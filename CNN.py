'''
The file is to construct a CNN model based on TextCNN, which has multiple convolutional blocks with different filter sizes   
'''

from DataGenerator import pad, generateData
from parameters import DEVICE, SBERT_VERSION, MAX_SENT_LENGTH, MAX_PARA_LENGTH
from parameters import MENU, SAVE_HISTORY, SAVE_MODEL, N_HIDDEN, CNN_WINDOWS, CNN_LR, EMB_SIZE, BATCH_SIZE, N_EPOCH
from ModelScore import ProduceAUC, plot_loss
import os
import pandas as pd
import numpy as np
import math
from tqdm import tqdm
from matplotlib import pyplot as plt
from transformers import AutoModel
import torch
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self, emb_size, max_n_sent, sent_length, n_hidden, windows):
        #
        super(CNNModel, self).__init__()  
        self.emb_size = emb_size
        self.max_n_sent = max_n_sent
        self.window_sizes = windows
        self.sent_length = sent_length
        self.n_hidden = n_hidden
        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv1d(in_channels = self.emb_size, out_channels = self.n_hidden, kernel_size = h, padding = 'same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size = 2, stride = 1),    #(self.max_n_sent-k)/s+1
            nn.Conv1d(in_channels = self.n_hidden, out_channels = self.n_hidden, kernel_size = h, padding = 'same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size = self.max_n_sent-1))
        for h in self.window_sizes])

    def forward(self, x1, x2):
        mid1 = torch.mean(x1, 2)      #from (batch_size, n_sentence, n_words, emb_size) to (batch_size, n_sentence, emb_size)-> average accross words(from word to sent)
        mid2 = torch.mean(x2, 2)
        Mid1 = mid1.permute(0, 2, 1)    #from (batch_size, n_sentence, emb_size) to (batch_size, emb_size, n_sentence)   #the conv is for sent to para
        Mid2 = mid2.permute(0, 2, 1)
        layer1 = [conv(Mid1) for conv in self.convs]  #(batch_size, n_hidden, 1) * len(self.window_sizes)
        layer2 = [conv(Mid2) for conv in self.convs]
        Out1 = torch.cat(layer1, dim=1)    #(batch_size, n_hidden*len(self.window_sizes), 1) 
        Out2 = torch.cat(layer2, dim=1)
        out1 = Out1.view(-1, Out1.size(1))
        out2 = Out2.view(-1, Out2.size(1))      #(batch_size, #hidden*len(self.window_sizes))
        y_hat = nn.CosineSimilarity(dim = 1)(out1, out2) #shape=(batch_size)
        y_hat = torch.clamp(y_hat, 0, 1)   #cos_sim can be negative
        return y_hat

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

    config = {"emb_size":EMB_SIZE,"max_n_sent":MAX_PARA_LENGTH,"sent_length": MAX_SENT_LENGTH,"n_hidden":N_HIDDEN,"windows":CNN_WINDOWS}
    CNNmodel = CNNModel(**config).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(CNNmodel.parameters(), lr = CNN_LR)

    if option == '1':    #new model
        history = {'train loss':[], 'val loss':[]}
        train(CNNmodel, encoder, criterion, optimizer, train_generator, val_generator, history, model_dir, hist_dir)
        plot_loss(history)
    
    elif option == '2':   #continue paused training
        checkpoint = torch.load(model_dir)
        CNNmodel.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        history = torch.load(hist_dir)
        val_loss = checkpoint['validation_loss']
        CNNmodel.train()
        train(CNNmodel, encoder, criterion, optimizer, train_generator, val_generator, history, model_dir, hist_dir, val_loss)
        plot_loss(history)
    
    else:    #evaluation
        checkpoint = torch.load(model_dir)
        CNNmodel.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        val_loss = checkpoint['validation_loss']

        CNNmodel.eval()
        eval(CNNmodel, encoder, test_generator)
