'''
The baseline model of average pooling. 
There is no training involved, hence only evaluation methods involved here.
Also generate a .csv file for model performance (output similarity scores for each test instance) logging.
'''

from DataGenerator import Dataset, cust_collate, pad
from parameters import DEVICE, TEST_PARAM, SBERT_VERSION, MAX_SENT_LENGTH, MAX_PARA_LENGTH, BATCH_SIZE, EMB_SIZE
from ModelScore import ProduceAUC
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from transformers import AutoModel
import torch
import torch.nn as nn

def generateData(batch_size):
    # To create a generator for test data
    # from the DataGenerator.py
    
    testset = Dataset('test')
    test_generator = torch.utils.data.DataLoader(testset, collate_fn=cust_collate, **TEST_PARAM)

    return len(testset), test_generator

def baseline_oper(ts):
    # Baseline operation: average pooling
    # [BATCH_SIZE, MAX_PARA_LENGTH, EMB_SIZE] --> [BATCH_SIZE, EMB_SIZE]
    
    return torch.mean(ts, 1)

def eval(encoder_model, record, generator, score_df):
    '''
    Evaluation of the baseline model.
    
    @ encoder_model (model): the sentence encoder (SBERT) to encode the sentence tokens
    @ record (string): the name to be recorded in the model performance log (column name)
    @ generator (Dataset object): for loading minibatches of testing data
    @ score_df (dataframe): for performance logging
    '''
    for ids, ids_b, label, id in tqdm(generator):
        # padding the paragraphs, so each paragraph contains exactly MAX_PARA_LENGTH sentences
        pad(ids, MAX_PARA_LENGTH, MAX_SENT_LENGTH)
        pad(ids_b, MAX_PARA_LENGTH, MAX_SENT_LENGTH)

        # adjustment of tensor dimension to suit into the SBERT encoder
        idst = torch.as_tensor(ids).view(BATCH_SIZE * MAX_PARA_LENGTH, -1).to(DEVICE)
        ids_bt = torch.as_tensor(ids_b).view(BATCH_SIZE * MAX_PARA_LENGTH, -1).to(DEVICE)
        
        with torch.no_grad():
            # sentence encoding
            emb = encoder_model(idst).last_hidden_state.view(-1, MAX_PARA_LENGTH, MAX_SENT_LENGTH, EMB_SIZE).cpu()
            emb_b = encoder_model(ids_bt).last_hidden_state.view(-1, MAX_PARA_LENGTH, MAX_SENT_LENGTH, EMB_SIZE).cpu()
        
        emb_m = torch.mean(emb, 2)    # [BATCH_SIZE, MAX_PARA_LENGTH, MAX_SENT_LENGTH, EMB_SIZE] --> [BATCH_SIZE, MAX_PARA_LENGTH, EMB_SIZE]
        emb_m = baseline_oper(emb_m)

        emb_b_m = torch.mean(emb_b, 2)
        emb_b_m = baseline_oper(emb_b_m)

        s = nn.CosineSimilarity(dim = 1)(emb_m, emb_b_m)
        s = torch.clamp(s, 0, 1)
        
        for i in range(len(id)):
            # log the model output into the dataframe for recording purpose.
            score_df[record][id[i]] = s.detach().numpy()[i]

def Baseline_eval():
    '''
    To initialise the parameters for the baseline model evaluaiton.
    After model output being logged, it will call the ProduceAUC() method to calculate AUROC score and display the ROC curve.
    '''
    record = input('Enter new record name:')
    length, data_generator = generateData(BATCH_SIZE)
    encoder = AutoModel.from_pretrained(SBERT_VERSION).to(DEVICE)
    score_df = pd.DataFrame(index = range(0, length), columns = [record], dtype = float)

    eval(encoder, data_generator, score_df)
    torch.save(score_df, 'score.pt')

    ProduceAUC()

if __name__ == '__main__':
    # main method.
    Baseline_eval()
