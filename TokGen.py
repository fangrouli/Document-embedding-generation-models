'''
Encoding of the paragraphs using the SBERT encoder.
Cache the generated embeddings in respective .pt files.
'''

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from transformers import AutoTokenizer
from parameters import SBERT_VERSION, MAX_SENT_LENGTH

import torch
import nltk
nltk.download('punkt')

pd.options.mode.chained_assignment = None
plt.switch_backend('agg')

def save_labels(df, path):
    '''
    Cache the labels of the instances into a .pt file for training and testing.
    
    @ df (dataframe): The dataframe that stores the labels.
    @ path (string): The directory to store the labels.
    '''
    ls = []
    for i in range(len(df)):
        ls.append(df['label'][i])
    torch.save(ls, path)

def read_write(old_df, new_df,
               generator,
               idx,
               max_length):  
    '''
    Tokenize and encode the data into embeddings.
    Stores the embeddings in another dataframe.
    
    @ old_df (dataframe): The dataframe that stores the original data.
    @ new_df (dataframe): The dataframe that stores the encoded data.
    @ generator (model): The pre-trained SBERT encoder.
    @ idx (int): The index of the current instance that is embedded.
    @ max_length (int): The MAX_SENT_LENGTH.
    '''

    s_ls = nltk.sent_tokenize(old_df['text'][idx])
    s_b_ls = nltk.sent_tokenize(old_df['text_b'][idx])

    w_ls = []
    for sentence in s_ls:
        ids = generator.encode(sentence,
                               padding = 'max_length',
                               truncation = True,
                               max_length = max_length)
        w_ls.append(ids)
        
    new_df['text'][idx] = w_ls

    w_ls = []
    for sentence in s_b_ls:
        ids = generator.encode(sentence,
                               padding = 'max_length',
                               truncation = True,
                               max_length = max_length)
        w_ls.append(ids)
        
    new_df['text_b'][idx] = w_ls


def tokenize(df, path):
    '''
    Started the encoding process, save the dataframe with embeddings in a .pt file for caching.
    
    @ df (dataframe): The dataframe that stores original data.
    @ path (str): The path for the dataframe with embeddings to store.
    '''
    text_df = pd.DataFrame(np.nan,
                           index=range(len(df)),
                           columns=['text', 'text_b'],
                           dtype = 'object')

    tokenizer = AutoTokenizer.from_pretrained(SBERT_VERSION, from_pt=True)

    for i in range(len(df)):
        read_write(df, text_df, tokenizer, i, MAX_SENT_LENGTH)

    torch.save(text_df, path)


if __name__ == '__main__':
    train = pd.read_csv('Archived/train.csv', index_col = 0)
    test = pd.read_csv('Archived/test.csv', index_col = 0)

    print(train.head())

    save_labels(train, 'train_labels.pt')
    save_labels(test, 'test_labels.pt')

    tokenize(train, 'train_tok.pt')
    tokenize(test, 'test_tok.pt')
    print('=== END ===')
    

    
