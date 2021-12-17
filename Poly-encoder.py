'''
Defines the poly-encoder model, and its training, evaluation methods.
'''

from DataGenerator import pad, generateData
from parameters import DEVICE, SBERT_VERSION, MAX_SENT_LENGTH, MAX_PARA_LENGTH
from parameters import N_EPOCH, POLY_M, POLY_LR, EMB_SIZE, BATCH_SIZE
from parameters import MENU, SAVE_HISTORY, SAVE_MODEL
from ModelScore import ProduceAUC, plot_loss
import numpy as np
from tqdm import tqdm
from transformers import AutoModel
import torch
from torch import nn
import torch.nn.functional as F

# Define a global cache space to store the similarity result of poly-encoder, as the model output is the loss instead of similarity.
result = [0, 0]

def get_result(layer_name):
    ''' The hook method used in poly-encoder model to extract the generated similarity score from the specified layer.
    @ layer_name (str): The layer name that output similarity score.
    '''
    def hook1(model, input, output):
        global result
        t = torch.clamp(output, 0, 1).detach().cpu().numpy()
        for i in range(output.shape[0]):
            result[i] = t[i]
    return hook1

#Reference: https://github.com/chijames/Poly-Encoder/blob/master/encoder.py
class PolyEncoder(nn.Module):
    def __init__(self, poly_m, emb_size, max_n_sent):
        ''' Initilisation of the poly-encoder model
        
        @ poly_m (int): The shape of the vectors in poly_m matrix used in the model, POLY_M.
        @ emb_size (int): Shape of the word embedding, EMB_SIZE.
        @ max_n_sent (int): Number of sentences in the paragraph, MAX_PARA_LENGTH.
        '''
        super().__init__()
        self.poly_m = poly_m
        self.emb_size = emb_size
        self.max_n_sent = max_n_sent
        self.poly_code_embeddings = nn.Embedding(self.poly_m, emb_size)
        torch.nn.init.normal_(self.poly_code_embeddings.weight, emb_size ** -0.5)
        self.decoder = nn.CosineSimilarity(dim = 1)

    def dot_attention(self, q, k, v):
        ''' The dot attention layer used in poly-encoder.
        
        @ q (tensor): [bs, poly_m, dim] or [bs, res_cnt, dim].
        @ k = v (tensor): [bs, length, dim] or [bs, poly_m, dim].
        '''
        attn_weights = torch.matmul(q, k.transpose(2, 1)) # [bs, poly_m, length]
        attn_weights = F.softmax(attn_weights, -1)
        output = torch.matmul(attn_weights, v) # [bs, poly_m, dim]
        return output

    def forward(self, emb, emb_b, labels=None):
        emb = torch.mean(emb, 2)
        emb_b = torch.mean(emb_b, 2) # (batch_size, n_sentence, n_words, emb_size) --> (batch_size, n_sentence, emb_size)

        batch_size = emb.shape[0]
        dim = self.emb_size
        res_cnt = 1

        poly_code_ids = torch.arange(self.poly_m, dtype=torch.long).to(emb.device)
        poly_code_ids = poly_code_ids.unsqueeze(0).expand(batch_size, self.poly_m)
        poly_codes = self.poly_code_embeddings(poly_code_ids) # [bs, poly_m, dim]

        # context encoder
        cont_embs = self.dot_attention(poly_codes, emb, emb) # [bs, poly_m, dim]

        # merge (global interaction)
        if batch_size == 1:
            ctx_emb = self.dot_attention(emb_b, cont_embs, cont_embs) # [bs, length, dim]
        else:
            ctx_emb = self.dot_attention(emb_b, cont_embs, cont_embs).squeeze()
            
        t_ctx = ctx_emb.view(-1, self.emb_size * self.max_n_sent)
        t_res = emb_b.view(-1, self.emb_size * self.max_n_sent)
        cossim = self.decoder(t_ctx, t_res)
        dot_product = (ctx_emb * emb_b).sum(-1) # [bs, length]
        if labels == None:
            return dot_product
        else:
            mask = torch.eye(batch_size, self.max_n_sent).to(emb_b.device) # [bs, length]
            loss = F.log_softmax(dot_product, dim=-1) * mask
            loss = (-loss.sum(dim=1)).mean()
            return loss

def train(model, encoder, optimizer, train_generator, val_generator, history, model_dir, hist_dir, prev_ep_val_loss = 100):
    ''' Training and validaiton of the model
    
    @ model (PolyEncoder object): Initialized poly-encoder model to be trained.
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

            y_true = torch.as_tensor(label, dtype = torch.float32).to(DEVICE)
            train_loss = model(emb, emb_b, y_true).to(DEVICE)

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

                y_true = torch.as_tensor(label, dtype = torch.float32).to(DEVICE)
                val_loss = model(emb, emb_b, y_true).to(DEVICE)

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

def eval(model, encoder, test_generator):
    ''' Evaluation of the model
    
    @ model (PolyEncoder object): Trained poly-encoder model to be evaluation.
    @ encoder (model): Pre-trained SBERT sentence encoder.
    @ test_generator (Dataset object): The mini-batch generator for testing.
    '''
    global result
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
            
            y_true = torch.as_tensor(label, dtype = torch.float32).to(DEVICE)
            test_loss = model(emb, emb_b, y_true).to(DEVICE)

        for i in range(len(id)):
            score_df[record][id[i]] = result[i]

    torch.save(score_df, 'score.pt')
    ProduceAUC()

if __name__ == "__main__":
    train_generator, val_generator, test_generator = generateData(BATCH_SIZE)
    encoder = AutoModel.from_pretrained(SBERT_VERSION).to(DEVICE)

    option, model_dir, hist_dir = MENU()

    config = {'poly_m': POLY_M, 'emb_size': EMB_SIZE, 'max_n_sent': MAX_PARA_LENGTH}
    poly_encoder = PolyEncoder(**config).to(DEVICE)
    optimizer = torch.optim.Adam(poly_encoder.parameters(), lr = POLY_LR)
    poly_encoder.decoder.register_forward_hook(get_result('decoder'))

    # Train the initialized new model form start
    if option == '1':
        history = {'train loss':[], 'val loss':[]}
        train(poly_encoder, encoder, optimizer, 
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
        train(poly_encoder, encoder, optimizer, train_generator, 
              val_generator, history, model_dir, hist_dir, val_loss)
        plot_loss(history)
    
    # Load and evaluation of a trained model
    else:
        checkpoint = torch.load(model_dir)
        poly_encoder.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        val_loss = checkpoint['validation_loss']

        poly_encoder.eval()
        eval(poly_encoder, poly_encoder, encoder, test_generator)
