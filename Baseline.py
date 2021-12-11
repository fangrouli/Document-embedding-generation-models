from DataGenerator import Dataset, cust_collate, pad
from parameters import DEVICE, TEST_PARAM, SBERT_VERSION, MAX_SENT_LENGTH, MAX_PARA_LENGTH, BATCH_SIZE, EMB_SIZE
from ModelScore import ProduceAUC
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from transformers import AutoModel
import torch
import torch.nn as nn

plt.switch_backend('agg')

def generateData(batch_size):

    testset = Dataset('test')
    test_generator = torch.utils.data.DataLoader(testset, collate_fn=cust_collate, **TEST_PARAM)

    return len(testset), test_generator

def baseline_oper(ts):
    return torch.mean(ts, 1)

def eval(encoder_model, record, generator, score_df):
    for ids, ids_b, label, id in tqdm(generator):
        pad(ids, MAX_PARA_LENGTH, MAX_SENT_LENGTH)
        pad(ids_b, MAX_PARA_LENGTH, MAX_SENT_LENGTH)

        idst = torch.as_tensor(ids).view(BATCH_SIZE * MAX_PARA_LENGTH, -1).to(DEVICE)
        ids_bt = torch.as_tensor(ids_b).view(BATCH_SIZE * MAX_PARA_LENGTH, -1).to(DEVICE)
        
        with torch.no_grad():
            emb = encoder_model(idst).last_hidden_state.view(-1, MAX_PARA_LENGTH, MAX_SENT_LENGTH, EMB_SIZE).cpu()
            emb_b = encoder_model(ids_bt).last_hidden_state.view(-1, MAX_PARA_LENGTH, MAX_SENT_LENGTH, EMB_SIZE).cpu()

        emb_m = torch.mean(emb, 2)
        emb_m = baseline_oper(emb_m)

        emb_b_m = torch.mean(emb_b, 2)
        emb_b_m = baseline_oper(emb_b_m)

        s = nn.CosineSimilarity(dim = 1)(emb_m, emb_b_m)
        s = torch.clamp(s, 0, 1)
        
        for i in range(len(id)):
            score_df[record][id[i]] = s.detach().numpy()[i]

def Baseline_eval():
    record = input('Enter new record name:')
    length, data_generator = generateData(BATCH_SIZE)
    encoder = AutoModel.from_pretrained(SBERT_VERSION).to(DEVICE)
    score_df = pd.DataFrame(index = range(0, length), columns = [record], dtype = float)

    eval(encoder, data_generator, score_df)
    torch.save(score_df, 'score.pt')

    ProduceAUC()

if __name__ == '__main__':
    Baseline_eval()
