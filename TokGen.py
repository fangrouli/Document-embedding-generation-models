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
    ls = []
    for i in range(len(df)):
        ls.append(df['label'][i])
    torch.save(ls, path)

def read_write(old_df, new_df,
               generator,
               idx,
               max_length):  

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
    

    
