'''
The CNN model developed based on TextCNN. The model input is a pair of sentence embeddings, output is the cosine-similarity of the pair.
Training is conducted with model checkpoints and early-stopping.
Evaluation will save the model output of the test dataset into the score logging .csv file, and generate the AUROC score.
'''

from DataGenerator import pad, generateData
from parameters import DEVICE, SBERT_VERSION, MAX_SENT_LENGTH, MAX_PARA_LENGTH
from parameters import MENU, SAVE_HISTORY, SAVE_MODEL, N_HIDDEN, CNN_WINDOWS, CNN_LR, EMB_SIZE, BATCH_SIZE, N_EPOCH
from ModelScore import ProduceAUC, plot_loss
import numpy as np
from tqdm import tqdm
from transformers import AutoModel
import torch
import torch.nn as nn

class CNNModel(nn.Module):
    '''Construction of CNN model.'''
    
    def __init__(self, emb_size, max_n_sent, sent_length, n_hidden, windows):
        ''' Initialisation of model
        
        @ emb_size (int): Shape of the word embedding, EMB_SIZE.
        @ max_n_sent (int): Number of sentences in the paragraph, MAX_PARA_LENGTH.
        @ sent_length (int): Number of words in the sentence, MAX_SENT_LENGTH.
        @ n_hidden (int): Number of hidden units (layer output channels) in Conv. layers, N_HIDDEN.
        @ windows (list): A list of integers, for the different filter sizes for different Conv. blocks, CNN_WINDOWS.
        '''
        super(CNNModel, self).__init__()  
        self.emb_size = emb_size
        self.max_n_sent = max_n_sent
        self.window_sizes = windows
        self.sent_length = sent_length
        self.n_hidden = n_hidden
        
        # model archiecture 
        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv1d(in_channels = self.emb_size, out_channels = self.n_hidden, kernel_size = h, padding = 'same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size = 2, stride = 1),    #(self.max_n_sent-k)/s+1
            nn.Conv1d(in_channels = self.n_hidden, out_channels = self.n_hidden, kernel_size = h, padding = 'same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size = self.max_n_sent-1))
        for h in self.window_sizes])

    def forward(self, x1, x2):
        mid1 = torch.mean(x1, 2)      # (batch_size, n_sentence, n_words, emb_size) --> (batch_size, n_sentence, emb_size)
        mid2 = torch.mean(x2, 2)
        
        Mid1 = mid1.permute(0, 2, 1)    # (batch_size, n_sentence, emb_size) --> (batch_size, emb_size, n_sentence)   
        Mid2 = mid2.permute(0, 2, 1)
        
        layer1 = [conv(Mid1) for conv in self.convs]  #(batch_size, n_hidden, 1) * len(self.window_sizes)
        layer2 = [conv(Mid2) for conv in self.convs]
        
        Out1 = torch.cat(layer1, dim=1)    #(batch_size, n_hidden*len(self.window_sizes), 1) 
        Out2 = torch.cat(layer2, dim=1)
        
        out1 = Out1.view(-1, Out1.size(1))
        out2 = Out2.view(-1, Out2.size(1))      #(batch_size, #hidden*len(self.window_sizes))
        
        y_hat = nn.CosineSimilarity(dim = 1)(out1, out2)     #shape=(batch_size)
        y_hat = torch.clamp(y_hat, 0, 1)
        return y_hat

def train(model, encoder, criterion, optimizer, train_generator, val_generator, history, model_dir, hist_dir, prev_ep_val_loss = 100):
    ''' Training and validaiton of the model
    
    @ model (CNNModel object): Initialized CNN model to be trained.
    @ encoder (model): Pre-trained SBERT sentence encoder.
    @ criterion (loss funtion): The loss function of the model.
    @ optimizer (optimizer object): The optimizer of the model.
    @ train_generator / val_generator (Dataset object): The mini-batch generator for more efficient training.
    @ history (dictionary): For logging of the training performance, including training loss and validation loss.
    @ model_dir (str): Directory for storing of the model checkpoints.
    @ hist_dir (str): Directory for storing of the training history, in case of resumed training.
    @ prev_ep_val_loss (float): In case of resumed training, for continuation of early-stopping.
    '''
    num_epoch = N_EPOCH         # number of max trianing epochs
    patience = 2                # number of early-stopping patience
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
            
            # encoding of tokens into embeddings
            with torch.no_grad():
                emb = encoder(idst).last_hidden_state.view(-1, MAX_PARA_LENGTH, MAX_SENT_LENGTH, EMB_SIZE)
                emb_b = encoder(ids_bt).last_hidden_state.view(-1, MAX_PARA_LENGTH, MAX_SENT_LENGTH, EMB_SIZE)

            y_pred = model(emb, emb_b).to(DEVICE)   # shape = (batch_size)
            y_true = torch.as_tensor(label, dtype = torch.float32).to(DEVICE)
            train_loss = criterion(y_pred, y_true)

            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_epoch_loss += y_pred.shape[0] * train_loss.item()
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

                y_pred = model(emb, emb_b).to(DEVICE)   # shape = (batch_size)
                y_true = torch.as_tensor(label, dtype = torch.float32).to(DEVICE)
                val_loss = criterion(y_pred, y_true)

            val_epoch_loss += y_pred.shape[0] * val_loss.item()
            instance_cnt += len(id)

        val_epoch_loss /= instance_cnt
        history['val loss'].append(val_epoch_loss)
        print(f'epoch: {epoch}, training loss = {train_epoch_loss:.4f}, validation loss = {val_epoch_loss:.4f}')
        SAVE_HISTORY(history, hist_dir)

        #early-stopping, patience = 2, validation loss
        if val_epoch_loss < prev_ep_val_loss:
            print(f'Improved from previous epoch ({prev_ep_val_loss:.4f}), model checkpoint saved to {model_dir}.')
            earlystop_cnt = 0
            
            # model checkpoint
            SAVE_MODEL(model, optimizer, model_dir, val_epoch_loss)
            prev_ep_val_loss = val_epoch_loss
        else:
            if earlystop_cnt < patience: # 1st epoch (patience suffering)
                print(f'No improvement from previous epoch ({prev_ep_val_loss:.4f})')
                earlystop_cnt += 1
            else:
                print(f'No improvement from previous epoch ({prev_ep_val_loss:.4f})')
                break

def eval(model, encoder, test_generator):
    ''' Evaluation of the model
    
    @ model (CNNModel object): Trained CNN model to be evaluated.
    @ encoder (model): Pre-trained SBERT sentence encoder.
    @ test_generator (Dataset object): The mini-batch generator for testing.
    '''
    
    score_df = torch.load('score.pt')
    record = input('Enter new record name:')    # Enter the recording name of the model for logging purpose.
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
    
    # Generating AUROC score and ROC curve.
    ProduceAUC()

if __name__ == "__main__":
    train_generator, val_generator, test_generator = generateData(BATCH_SIZE)
    encoder = AutoModel.from_pretrained(SBERT_VERSION).to(DEVICE)

    option, model_dir, hist_dir = MENU()

    config = {"emb_size":EMB_SIZE,"max_n_sent":MAX_PARA_LENGTH,"sent_length": MAX_SENT_LENGTH,"n_hidden":N_HIDDEN,"windows":CNN_WINDOWS}
    CNNmodel = CNNModel(**config).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(CNNmodel.parameters(), lr = CNN_LR)

    # Initialize and train a new model
    if option == '1':    
        history = {'train loss':[], 'val loss':[]}
        train(CNNmodel, encoder, criterion, optimizer, train_generator, val_generator, history, model_dir, hist_dir)
        plot_loss(history)
    
    # Resume the paused training of a cached model
    elif option == '2':   
        checkpoint = torch.load(model_dir)
        CNNmodel.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        history = torch.load(hist_dir)
        val_loss = checkpoint['validation_loss']
        CNNmodel.train()
        train(CNNmodel, encoder, criterion, optimizer, train_generator, val_generator, history, model_dir, hist_dir, val_loss)
        plot_loss(history)
    
    # Evaluation of a cached model
    else:    
        checkpoint = torch.load(model_dir)
        CNNmodel.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        val_loss = checkpoint['validation_loss']

        CNNmodel.eval()
        eval(CNNmodel, encoder, test_generator)
