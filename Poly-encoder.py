from DataGenerator import pad, generateData
from parameters import DEVICE, SBERT_VERSION, MAX_SENT_LENGTH, MAX_PARA_LENGTH
from parameters import N_EPOCH, POLY_M, POLY_LR, EMB_SIZE, BATCH_SIZE
from parameters import MENU, SAVE_HISTORY, SAVE_MODEL
from ModelScore import ProduceAUC, plot_loss
import os
import pandas as pd
import numpy as np
import math
from tqdm import tqdm
from matplotlib import pyplot as plt
from transformers import AutoModel
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
result = [0, 0]

def get_result(layer_name):
    def hook1(model, input, output):
        global result
        t = torch.clamp(output, 0, 1).detach().cpu().numpy()
        for i in range(output.shape[0]):
            result[i] = t[i]
    return hook1

#Reference: https://github.com/chijames/Poly-Encoder/blob/master/encoder.py
class PolyEncoder(nn.Module):
    def __init__(self, poly_m, emb_size, max_n_sent):
        super().__init__()
        #self.bert = kwargs['bert']
        self.poly_m = poly_m
        self.emb_size = emb_size
        self.max_n_sent = max_n_sent
        self.poly_code_embeddings = nn.Embedding(self.poly_m, emb_size)
        torch.nn.init.normal_(self.poly_code_embeddings.weight, emb_size ** -0.5)
        self.decoder = nn.CosineSimilarity(dim = 1)

    def dot_attention(self, q, k, v):
        # para_emb: [bs, max_n_sent, dim]
        # query: [bs, max_n_sent, dim] or [bs, poly_m, dim]
        # q: [bs, max_n_sent, dim] or [bs, poly_m, dim]
        # k = v: [bs, max_n_sent, dim]

        attn_weights = torch.matmul(q, k.transpose(2, 1)) # [bs, poly_m, length]
        attn_weights = F.softmax(attn_weights, -1)
        output = torch.matmul(attn_weights, v) # [bs, poly_m, dim]
        return output

    def forward(self, emb, emb_b, labels=None): #[bs, n_sent, n_word, dim]
        emb = torch.mean(emb, 2)
        emb_b = torch.mean(emb_b, 2) #[bs, n_sent, dim]

        batch_size = emb.shape[0]
        dim = self.emb_size
        res_cnt = 1

        poly_code_ids = torch.arange(self.poly_m, dtype=torch.long).to(emb.device)
        poly_code_ids = poly_code_ids.unsqueeze(0).expand(batch_size, self.poly_m)
        poly_codes = self.poly_code_embeddings(poly_code_ids) # [bs, poly_m, dim]

        # context encoder
        cont_embs = self.dot_attention(poly_codes, emb, emb) # [bs, poly_m, dim]

        # merge
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

        #if (epoch+1) % 5 == 0:
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

    if option == '1':    #new model
        history = {'train loss':[], 'val loss':[]}
        train(poly_encoder, encoder, optimizer, 
              train_generator, val_generator, history, model_dir, hist_dir)
        plot_loss(history)
    
    elif option == '2':   #continue paused training
        checkpoint = torch.load(model_dir)
        poly_encoder.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        history = torch.load(hist_dir)
        val_loss = checkpoint['validation_loss']
        poly_encoder.train()
        train(poly_encoder, encoder, optimizer, train_generator, 
              val_generator, history, model_dir, hist_dir, val_loss)
        plot_loss(history)
    
    else:    #evaluation
        checkpoint = torch.load(model_dir)
        poly_encoder.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        val_loss = checkpoint['validation_loss']

        poly_encoder.eval()
        eval(poly_encoder, poly_encoder, encoder, test_generator)
