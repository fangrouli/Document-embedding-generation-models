from DataGenerator import pad, generateData
from parameters import DEVICE, SBERT_VERSION, MAX_SENT_LENGTH, MAX_PARA_LENGTH, N_HEAD, TRANS_DROPOUT, TRANS_LAYER, TRANS_LR
from parameters import MENU, SAVE_HISTORY, SAVE_MODEL, TRANS_N_HIDDEN, EMB_SIZE, BATCH_SIZE, N_EPOCH
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
from torch.nn import TransformerEncoder, TransformerEncoderLayer

#Reference: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class TransformerModel(nn.Module):
    def __init__(self, emb_size, max_n_sent, n_hidden, n_head, n_layers, dropout):
        super().__init__()
        self.model_type = 'Transformer'
        self.emb_size = emb_size
        self.pos_encoder = PositionalEncoding(emb_size, max_n_sent, dropout)

        encoder_layers = TransformerEncoderLayer(emb_size, n_head, n_hidden, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        self.decoder = nn.CosineSimilarity(dim = 1)
        self.pooling = nn.MaxPool1d(kernel_size = max_n_sent)
        #self.pooling = nn.AvgPool1d(kernel_size = max_n_sent)

    def forward(self, x1, x2) -> Tensor:
        # x1, x2: Tensor, shape [batch_size, n_sentence, n_words, emb_size]
        # output: similarity score

        mid1 = torch.mean(x1, 2)      #from (batch_size, n_sentence, n_words, emb_size) to (batch_size, n_sentence, emb_size)
        mid2 = torch.mean(x2, 2)

        Mid1 = mid1.permute(1, 0, 2)    #from (batch_size, n_sentence, emb_size) to (n_sentence, batch_size, emb_size)
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
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_n_sent).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2) * (-math.log(10000.0) / emb_size))
        pe = torch.zeros(max_n_sent, 1, emb_size)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: Tensor, shape [n_sentence, batch_size, emb_size]

        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

def train(model, encoder, criterion, optimizer, train_generator, val_generator, history, model_dir, hist_dir, prev_ep_val_loss = 100):
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

            y_pred = model(emb, emb_b).to(DEVICE)   #shape = (batch_size)
            y_true = torch.as_tensor(label, dtype = torch.float32).to(DEVICE)
            train_loss = criterion(y_pred, y_true)

            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

            train_epoch_loss += y_pred.shape[0] * train_loss.item()
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

                y_pred = model(emb, emb_b).to(DEVICE)   #shape = (batch_size)
                y_true = torch.as_tensor(label, dtype = torch.float32).to(DEVICE)
                val_loss = criterion(y_pred, y_true)

            val_epoch_loss += y_pred.shape[0] * val_loss.item()
            instance_cnt += len(id)

        val_epoch_loss /= instance_cnt
        history['val loss'].append(val_epoch_loss)
        print(f'epoch: {epoch}, training loss = {train_epoch_loss:.4f}, validation loss = {val_epoch_loss:.4f}')
        SAVE_HISTORY(history, hist_dir)

        #early stop, patience = 2, validation loss
        if val_epoch_loss < prev_ep_val_loss:
            print(f'Improved from previous epoch ({prev_ep_val_loss:.4f}), model checkpoint saved to {model_dir}.')
            earlystop_cnt = 0
            SAVE_MODEL(model, optimizer, model_dir, val_epoch_loss)
            prev_ep_val_loss = val_epoch_loss
        else:
            if earlystop_cnt < patience: #1st epoch
                print(f'No improvement from previous epoch ({prev_ep_val_loss:.4f})')#, model checkpoint saved to {path}.')
                earlystop_cnt += 1
            else:
                print(f'No improvement from previous epoch ({prev_ep_val_loss:.4f})')#, model checkpoint saved to {path}, exit training.')
                break

def eval(model, encoder, test_generator):
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

            y_pred = model(emb, emb_b).cpu()

        for i in range(len(id)):
            score_df[record][id[i]] = y_pred.detach().numpy()[i]

    torch.save(score_df, 'score.pt')
    ProduceAUC()

if __name__ == "__main__":
    train_generator, val_generator, test_generator = generateData(BATCH_SIZE)
    encoder = AutoModel.from_pretrained(SBERT_VERSION).to(DEVICE)

    option, model_dir, hist_dir = MENU()

    config = {"emb_size": EMB_SIZE, 
              "max_n_sent": MAX_PARA_LENGTH, 
              "n_hidden": TRANS_N_HIDDEN, 
              "n_head": N_HEAD, 
              "n_layers": TRANS_LAYER, 
              "dropout": TRANS_DROPOUT}

    transformer = TransformerModel(**config).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(transformer.parameters(), lr = TRANS_LR)

    if option == '1':    #new model
        history = {'train loss':[], 'val loss':[]}
        train(transformer, encoder, criterion, optimizer, train_generator, val_generator, history, model_dir, hist_dir)
        plot_loss(history)
    
    elif option == '2':   #continue paused training
        checkpoint = torch.load(model_dir)
        transformer.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        history = torch.load(hist_dir)
        val_loss = checkpoint['validation_loss']
        transformer.train()
        train(transformer, encoder, criterion, optimizer, train_generator, val_generator, history, model_dir, hist_dir, val_loss)
        plot_loss(history)
    
    else:    #evaluation
        checkpoint = torch.load(model_dir)
        transformer.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        val_loss = checkpoint['validation_loss']

        transformer.eval()
        eval(transformer, encoder, test_generator)
