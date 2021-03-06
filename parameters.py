'''
Defines all parameters and hyperparameters used in training and evaluation.
Defines the menu function, model checkpoint function and history caching function.
'''

import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SBERT_VERSION = "sentence-transformers/paraphrase-mpnet-base-v2"
MAX_SENT_LENGTH = 128
MAX_PARA_LENGTH = 8
BATCH_SIZE = 2
EMB_SIZE = 768

N_HIDDEN = 100
N_EPOCH = 20
CNN_WINDOWS = [2, 3]
CNN_LR = 0.0001

TRANS_LR = 0.0001
TRANS_N_HIDDEN = 50
N_HEAD = 2
TRANS_LAYER = 2
TRANS_DROPOUT = 0.2

POLY_M = 16
POLY_LR = 0.0001

TEST_PARAM = {'batch_size':BATCH_SIZE, 'shuffle': False}
TRAIN_PARAM = {'batch_size':BATCH_SIZE, 'shuffle': False}
VAL_PARAM = {'batch_size':BATCH_SIZE, 'shuffle': False}

def MENU():
    ''' The menu function. '''
    
    model_dir = ''
    hist_dir = ''
    print("Please select your option:")
    print("1. Train a new model.")
    print("2. Continue training the last model.")
    print("3. Evaluate the last model.")
    option = input('Your Option: ')
    model_dir = input('Model Directory:')
    if option == '3':
        return option, model_dir, hist_dir
    else:
        hist_dir = input('Training History Directory:')
        return option, model_dir, hist_dir

def SAVE_MODEL(mod, opt, dir, val_loss):
    ''' Model checkpoint function.
    
    @ mod (model object): The model to be saved.
    @ opt (optimizer object): The model optimizer to be saved.
    @ dir (str): The directory for the checkpoint to be savec.
    @ val_loss (float): The last epoch's validation loss for resume of the training.
    '''
    torch.save({'model_state_dict': mod.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'validation_loss': val_loss}, dir)
    
def SAVE_HISTORY(his, dir):
    ''' History chache.
    
    @ his (dictionary): The dictionary that caches the training and validaiton losses.
    @ dir (str): The directory to save the history.
    '''
    torch.save(his, dir)
