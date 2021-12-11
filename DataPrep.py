import pandas as pd
pd.options.mode.chained_assignment = None  #ignore the warning
from tqdm import tqdm
import regex as re

def import_train_set(direc):
    with open(direc) as file:
        content = pd.read_csv(file, sep = '\t')
    file.close()
    return content

def sent_clear(sentence):
    pat = re.compile(r'(figure\s)|(figs\.)|(fig\.)|(\d[a-zA-Z]\s)|(\s[a-zA-Z]\d)|(\d\.\d)|(\d)')
    sentence = pat.sub(r'', sentence)

    sentence = re.sub('e\.g\.','for example ',sentence)
    sentence = re.sub('i\.e\.','in other words ',sentence)

    sentence = re.sub('\.','. ',sentence)
    sentence = re.sub('\s+',' ',sentence)
    sentence.rstrip()
    return sentence

def remove_1_sent(df, col):
  to_be_removed = []
  for i in tqdm(range(len(df))):
    sent =  df[col][i]
    sent_ls = sent.split('.')

    for s in sent_ls.copy():
      if ' ' not in s or len(s) <= 5:
        sent_ls.remove(s)

    if len(sent_ls) <= 1:
      to_be_removed.append(i)
  return to_be_removed


def cleaning(df, final_dir):
    text_rem = remove_1_sent(df, 'text_b')
    df.drop(labels = text_rem, axis = 0, inplace = True)
    df.reset_index(inplace = True, drop = True)
    
    for i in tqdm(range(len(df))):
        df['text'][i] = sent_clear(df['text'][i])
        df['text_b'][i] = sent_clear(df['text_b'][i])
        
    df.to_csv(final_dir)


if __name__ == '__main__':
    train_dir = input("Enter the train file directory (.tsv):")
    test_dir = input("Enter the test file directory (.tsv):")

    print("\n......Loading Data......")

    train_df = import_train_set(train_dir)[['text', 'text_b', 'label']]
    test_df = import_train_set(test_dir)[['text', 'text_b', 'label']]

    print("Training Data:")
    print(train_df.head())

    print("\n\nTesting Data")
    print(test_df.head())

    print("\n......Clearing Training Data......")
    cleaning(train_df, 'train.csv')
    cleaning(test_df, 'test.csv')
    
