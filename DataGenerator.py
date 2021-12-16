'''
Create the Dataset object serving as a customised data generator for producing mini-batches.
Customized the generator and the collate function.
Defined the padding function.
'''

from matplotlib import pyplot as plt
import torch
from parameters import TEST_PARAM, TRAIN_PARAM, VAL_PARAM

plt.switch_backend('agg')

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name):
        '''
		Defined the data generator object.
		
		@ dataset_name (string): different mode. 'train', 'test' or 'validation'.
		'''
        if dataset_name == 'train':
            data_df = torch.load('train_tok.pt')
            label_df = torch.load('train_labels.pt')
        elif dataset_name == 'test':
            data_df = torch.load('test_tok.pt')
            label_df = torch.load('test_labels.pt')
        elif dataset_name == 'validation':
            data_df = torch.load('val_tok.pt')
            label_df = torch.load('val_labels.pt')

        self.x1 = data_df['text']
        self.x2 = data_df['text_b']
        self.y = label_df
        self.length = len(self.x1)

    def __len__(self):
        return self.length
      
    def __getitem__(self, index):
        ids = self.x1[index]
        ids_b = self.x2[index]
        lb = self.y[index]

        return ids, ids_b, lb, index

def cust_collate(batch):
	'''
	Defines the shape of the mini-batches from the data generator
	'''
    ids = [item[0] for item in batch]
    ids_b = [item[1] for item in batch]

    lb = [item[2] for item in batch]
    index = [item[3] for item in batch]

    return ids, ids_b, lb, index


def pad(ids, max_len, sent_len):
	'''
	The padding and truncation of the paragraphs, in order for them to have a universal shape for training models.
	Returns sentence tokens of shape [batch_size, sent_len, max_len]
	
	@ ids (list): The input tokens of the paragraph
	@ max_len (int): the fixed sentence length, defined in parameters.py 
	@ sent_len (int): the fixed paragraph length, defined in parameters.py 
	'''
    empty_ls = [1]*sent_len         #for pmb model, 1 represents empty
    batch = len(ids)
    for i in range(batch):
        diff = max_len - len(ids[i])
        if diff > 0:
            for j in range(diff):
                ids[i].append(empty_ls)
        if diff < 0:
            ids[i] = ids[i][:max_len]

def generateData(batch_size):
    testset = Dataset('test')
    test_generator = torch.utils.data.DataLoader(testset, collate_fn=cust_collate, **TEST_PARAM)
    training_set = Dataset('train')
    training_generator = torch.utils.data.DataLoader(training_set, collate_fn=cust_collate, **TRAIN_PARAM)
    validation_set = Dataset('validation')
    validation_generator = torch.utils.data.DataLoader(validation_set, collate_fn=cust_collate, **VAL_PARAM)
    return training_generator, validation_generator, test_generator
