'''
Defines the transformer-poly-encoder model. 
The poly-encoder model will use the paragraph embeddings generated by the trained transformer model from the transformer.py.
Global interactions in the poly-encoder will hopefully improve the accuracy of the similarity comparison.
'''

from DataGenerator import pad, generateData
from parameters import DEVICE, SBERT_VERSION, MAX_SENT_LENGTH, MAX_PARA_LENGTH, N_HEAD
from parameters import TRANS_DROPOUT, TRANS_LAYER, TRANS_LR, N_EPOCH, POLY_M, POLY_LR
from parameters import MENU, SAVE_HISTORY, SAVE_MODEL, TRANS_N_HIDDEN, EMB_SIZE, BATCH_SIZE
from ModelScore import ProduceAUC, plot_loss
import numpy as np
import math
from tqdm import tqdm
from transformers import AutoModel
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# Define a global cache space to store the paragraph embeddings generated by transformer and the similarity result of poly-encoder.
para_embs = {}
result = [0, 0]

#Reference: https://github.com/chijames/Poly-Encoder/blob/master/encoder.py
class PolyEncoder(nn.Module):
    def __init__(self, poly_m, emb_size):
        ''' Initilisation of the poly-encoder model
        
        @ poly_m (int): The shape of the vectors in poly_m matrix used in the model, POLY_M.
        @ emb_size (int): Shape of the word embedding, EMB_SIZE.
        '''
        super().__init__()
        self.poly_m = poly_m
        self.emb_size = emb_size
        self.poly_code_embeddings = nn.Embedding(self.poly_m, emb_size)
        torch.nn.init.normal_(self.poly_code_embeddings.weight, emb_size ** -0.5)
        self.decoder = nn.CosineSimilarity(dim = 2)

    def dot_attention(self, q, k, v):
        ''' The dot attention layer used in poly-encoder.
        
        @ q (tensor): [bs, poly_m, dim] or [bs, res_cnt, dim].
        @ k = v (tensor): [bs, length, dim] or [bs, poly_m, dim].
        '''
        attn_weights = torch.matmul(q, k.transpose(2, 1)) # [bs, poly_m, length]
        print(attn_weights)
        attn_weights = F.softmax(attn_weights, -1)
        print(attn_weights)
        output = torch.matmul(attn_weights, v) # [bs, poly_m, dim]
        return output

    def forward(self, context_emb, responses_emb, labels=None):
        context_emb = context_emb.view(-1, 1, self.emb_size)
        responses_emb = responses_emb.view(-1, 1, self.emb_size)    #[bs, 1, dim]
        batch_size, res_cnt, dim = context_emb.shape # res_cnt is 1 during training

        # context encoder
        poly_code_ids = torch.arange(self.poly_m, dtype=torch.long).to(context_emb.device)
        poly_code_ids = poly_code_ids.unsqueeze(0).expand(batch_size, self.poly_m)
        poly_codes = self.poly_code_embeddings(poly_code_ids)                       # [bs, poly_m, dim]
        cont_embs = self.dot_attention(poly_codes, context_emb, context_emb)        # [bs, poly_m, dim]

        # merge (Global interaction)
        if labels is not None:
            cand_emb = responses_emb.permute(1, 0, 2) # [1, bs, dim]
            cand_emb = cand_emb.expand(batch_size, batch_size, cand_emb.shape[2])           # [bs, bs, dim]
            if batch_size == 1:
                ctx_emb = self.dot_attention(cand_emb, cont_embs, cont_embs)
            else:
                ctx_emb = self.dot_attention(cand_emb, cont_embs, cont_embs).squeeze()      # [bs, bs, dim]

            cossim = self.decoder(ctx_emb, cand_emb)
            dot_product = (ctx_emb*cand_emb).sum(-1)                # [bs, bs]
            mask = torch.eye(batch_size).to(context_emb.device)     # [bs, bs]
            loss = F.log_softmax(dot_product, dim=-1) * mask
            loss = (-loss.sum(dim=1)).mean()
            return loss
        else:
            ctx_emb = self.dot_attention(responses_emb, cont_embs, cont_embs)   # [bs, res_cnt, dim]
            dot_product = (ctx_emb*responses_emb).sum(-1)
            return dot_product

class TransformerPoly(nn.Module):
    def __init__(self, emb_size, max_n_sent, n_hidden, n_head, n_layers, dropout):
        ''' Initialize of the transformer model, for loading of the trained transformer '''
        
        super().__init__()
        self.model_type = 'Transformer'
        self.emb_size = emb_size
        self.pos_encoder = PositionalEncoding(emb_size, max_n_sent, dropout)

        encoder_layers = TransformerEncoderLayer(emb_size, n_head, n_hidden, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        self.decoder = nn.CosineSimilarity(dim = 1)
        self.pooling = nn.MaxPool1d(kernel_size = max_n_sent)

    def forward(self, x1, x2) -> Tensor:
        # x1, x2 (tensor): shape [batch_size, n_sentence, n_words, emb_size]
        # output (tensor): similarity score

        mid1 = torch.mean(x1, 2)    #(batch_size, n_sentence, emb_size)-> average accross words(from word to sent)
        mid2 = torch.mean(x2, 2)

        Mid1 = mid1.permute(1, 0, 2)    #(n_sentence, batch_size, emb_size)
        Mid2 = mid2.permute(1, 0, 2)

        Mid1 = self.pos_encoder(Mid1)
        Mid2 = self.pos_encoder(Mid2)

        output1 = self.transformer_encoder(Mid1)   #(n_sentence, batch_size, emb_size)
        output2 = self.transformer_encoder(Mid2)

        output1 = output1.permute(1, 2, 0)      #(batch_size, emb_size, n_sentence)
        output2 = output2.permute(1, 2, 0)
        
        Out1 = self.pooling(output1) #(batch_size, emb_size, 1)
        Out2 = self.pooling(output2)

        out1 = Out1.view(-1, Out1.size(1))  #(batch_size, emb_size)
        out2 = Out2.view(-1, Out2.size(1))

        f_output = self.decoder(out1, out2)
        f_output = torch.clamp(f_output, 0, 1)
        return f_output

class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, max_n_sent, dropout):
        ''' The positional encoding of the transformer ''' 
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_n_sent).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2) * (-math.log(10000.0) / emb_size))
        pe = torch.zeros(max_n_sent, 1, emb_size)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x (tensor): shape [n_sentence, batch_size, emb_size]

        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


def get_activation(layer_name):
    ''' The hook method used in tranformer model to extract the generated paragraph embedding from the specified layer.
    @ layer_name (str): The layer name that output paragraph embeddings.
    '''
    def hook(model, input, output):
        ''' Cache the paragraph embeddings in the global dictionary. '''
        
        global para_embs
        para_embs['para'],  para_embs['para_b']= input[0], input[1]
    return hook

def get_result(layer_name):
    ''' The hook method used in poly-encoder model to extract the generated similarity score from the specified layer.
    @ layer_name (str): The layer name that output similarity score.
    '''
    def hook1(model, input, output):
        ''' Cache the similarity score in the global list. '''
        
        global result
        t = torch.mean(output, 1).detach().cpu().numpy()
        for i in range(output.shape[0]):
            result[i] = t[i]
    return hook1

def train(transformer, poly_encoder, encoder, optimizer, train_generator, val_generator, 
          history, model_dir, hist_dir, prev_ep_val_loss = 100):
    ''' Training and validaiton of the model
    
    @ transformer (TransformerPoly object): Trained transformer for paragraph embedding generation.
    @ poly_encoder (PolyEncoder object): Initialized poly-encoder model to be trained.
    @ encoder (model): Pre-trained SBERT sentence encoder.
    @ optimizer (optimizer object): The optimizer of the model.
    @ train_generator / val_generator (Dataset object): The mini-batch generator for more efficient training.
    @ history (dictionary): For logging of the training performance, including training loss and validation loss.
    @ model_dir (str): Directory for storing of the model checkpoints.
    @ hist_dir (str): Directory for storing of the training history, in case of resumed training.
    @ prev_ep_val_loss (float): In case of resumed training, for continuation of early-stopping.
    '''
    num_epoch = N_EPOCH
    patience = 2
    earlystop_cnt = 0

    for epoch in range(num_epoch):
        train_epoch_loss = 0.0
        val_epoch_loss = 0.0
        instance_cnt = 0
        for ids, ids_b, label, id in tqdm(train_generator):
            pad(ids, MAX_PARA_LENGTH, MAX_SENT_LENGTH)     
            pad(ids_b, MAX_PARA_LENGTH, MAX_SENT_LENGTH)      

            idst = torch.as_tensor(ids).view(BATCH_SIZE * MAX_PARA_LENGTH, -1).to(DEVICE)
            ids_bt = torch.as_tensor(ids_b).view(BATCH_SIZE * MAX_PARA_LENGTH, -1).to(DEVICE)
            
            with torch.no_grad():
                emb = encoder(idst).last_hidden_state.view(-1, MAX_PARA_LENGTH, MAX_SENT_LENGTH, EMB_SIZE)
                emb_b = encoder(ids_bt).last_hidden_state.view(-1, MAX_PARA_LENGTH, MAX_SENT_LENGTH, EMB_SIZE)
                temp = transformer(emb, emb_b).to(DEVICE)

            y_true = torch.as_tensor(label, dtype = torch.float32).to(DEVICE)
            train_loss = poly_encoder(para_embs['para'], para_embs['para_b'], y_true).to(DEVICE)

            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_epoch_loss += y_true.shape[0] * train_loss.item()
            instance_cnt += len(id)

        train_epoch_loss /= instance_cnt
        history['train loss'].append(train_epoch_loss)

        #validation
        instance_cnt = 0
        for ids, ids_b, label, id in tqdm(val_generator):
            pad(ids, MAX_PARA_LENGTH, MAX_SENT_LENGTH)
            pad(ids_b, MAX_PARA_LENGTH, MAX_SENT_LENGTH)

            idst = torch.as_tensor(ids).view(BATCH_SIZE * MAX_PARA_LENGTH, -1).to(DEVICE)
            ids_bt = torch.as_tensor(ids_b).view(BATCH_SIZE * MAX_PARA_LENGTH, -1).to(DEVICE)
            
            with torch.no_grad():
                emb = encoder(idst).last_hidden_state.view(-1, MAX_PARA_LENGTH, MAX_SENT_LENGTH, EMB_SIZE)
                emb_b = encoder(ids_bt).last_hidden_state.view(-1, MAX_PARA_LENGTH, MAX_SENT_LENGTH, EMB_SIZE)
                temp = transformer(emb, emb_b).to(DEVICE)

                y_true = torch.as_tensor(label, dtype = torch.float32).to(DEVICE)
                val_loss = poly_encoder(para_embs['para'], para_embs['para_b'], y_true).to(DEVICE)

            val_epoch_loss += y_true.shape[0] * val_loss.item()
            instance_cnt += len(id)

        val_epoch_loss /= instance_cnt
        history['val loss'].append(val_epoch_loss)
        print(f'epoch: {epoch}, training loss = {train_epoch_loss:.4f}, validation loss = {val_epoch_loss:.4f}')
        SAVE_HISTORY(history, hist_dir)

        #early stop, patience = 2, validation loss
        if val_epoch_loss < prev_ep_val_loss:
            print(f'Improved from previous epoch ({prev_ep_val_loss:.4f}), model checkpoint saved to {model_dir}.')
            earlystop_cnt = 0
            SAVE_MODEL(poly_encoder, optimizer, model_dir, val_epoch_loss)
            prev_ep_val_loss = val_epoch_loss
        else:
            if earlystop_cnt < patience: #1st epoch
                print(f'No improvement from previous epoch ({prev_ep_val_loss:.4f})')
                earlystop_cnt += 1
            else:
                print(f'No improvement from previous epoch ({prev_ep_val_loss:.4f})')
                break

def eval(transformer, poly_encoder, encoder, test_generator):
    ''' Evaluation of the model
    
    @ transformer (TransformerPoly object): Trained transformer for paragraph embedding generation.
    @ poly_encoder (PolyEncoder object): Trained poly-encoder model to be evaluation.
    @ encoder (model): Pre-trained SBERT sentence encoder.
    @ test_generator (Dataset object): The mini-batch generator for testing.
    '''
    global result, para_embs
    score_df = torch.load('score.pt')
    record = input('Enter new record name:')
    score_df[record] = np.nan

    for ids, ids_b, label, id in tqdm(test_generator):
        pad(ids, MAX_PARA_LENGTH, MAX_SENT_LENGTH)
        pad(ids_b, MAX_PARA_LENGTH, MAX_SENT_LENGTH)

        idst = torch.as_tensor(ids).view(BATCH_SIZE * MAX_PARA_LENGTH, -1).to(DEVICE)
        ids_bt = torch.as_tensor(ids_b).view(BATCH_SIZE * MAX_PARA_LENGTH, -1).to(DEVICE)
        
        with torch.no_grad():
            emb = encoder(idst).last_hidden_state.view(-1, MAX_PARA_LENGTH, MAX_SENT_LENGTH, EMB_SIZE)
            emb_b = encoder(ids_bt).last_hidden_state.view(-1, MAX_PARA_LENGTH, MAX_SENT_LENGTH, EMB_SIZE)
            temp = transformer(emb, emb_b).to(DEVICE)
            
            y_true = torch.as_tensor(label, dtype = torch.float32).to(DEVICE)
            test_loss = poly_encoder(para_embs['para'], para_embs['para_b'], y_true).to(DEVICE)

        for i in range(len(id)):
            score_df[record][id[i]] = result[i]

    torch.save(score_df, 'score.pt')
    ProduceAUC()

if __name__ == "__main__":
    train_generator, val_generator, test_generator = generateData(BATCH_SIZE)
    encoder = AutoModel.from_pretrained(SBERT_VERSION).to(DEVICE)

    option, model_dir, hist_dir = MENU()

    # Loading of the trained tranformer
    trans_config = {"emb_size": EMB_SIZE, 
                    "max_n_sent": MAX_PARA_LENGTH, 
                    "n_hidden": TRANS_N_HIDDEN, 
                    "n_head": N_HEAD, 
                    "n_layers": TRANS_LAYER, 
                    "dropout": TRANS_DROPOUT}

    transformer = TransformerPoly(**trans_config).to(DEVICE)
    cri = nn.BCELoss()
    opt = torch.optim.Adam(transformer.parameters(), lr = TRANS_LR)
    pretrans_dir = input("\nEnter the pre-trained transformer directory: ")
    checkpoint = torch.load(pretrans_dir)
    transformer.load_state_dict(checkpoint['model_state_dict'])
    
    # Register the hook to extract paragraph embeddings
    transformer.decoder.register_forward_hook(get_activation('decoder'))
    opt.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Freeze the transformer so it is not involved in training
    for param in transformer.parameters(): 
        param.requires_grad = False

    # Initialise the poly_encoder
    config = {'poly_m': POLY_M, 'emb_size': EMB_SIZE}
    poly_encoder = PolyEncoder(**config).to(DEVICE)
    optimizer = torch.optim.Adam(poly_encoder.parameters(), lr = POLY_LR)
    poly_encoder.decoder.register_forward_hook(get_result('decoder'))

    # Train the new model form start
    if option == '1':    
        history = {'train loss':[], 'val loss':[]}
        train(transformer, poly_encoder, encoder, optimizer, 
              train_generator, val_generator, history, model_dir, hist_dir)
        plot_loss(history)
    
    # Load and resume paused training of an existing model
    elif option == '2':
        checkpoint = torch.load(model_dir)
        poly_encoder.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        history = torch.load(hist_dir)
        val_loss = checkpoint['validation_loss']
        poly_encoder.train()
        train(transformer, poly_encoder, encoder, optimizer, train_generator, 
              val_generator, history, model_dir, hist_dir, val_loss)
        plot_loss(history)
    
    # Load and evaluation of a trained model
    else:
        checkpoint = torch.load(model_dir)
        poly_encoder.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        val_loss = checkpoint['validation_loss']

        poly_encoder.eval()
        eval(transformer, poly_encoder, encoder, test_generator)
