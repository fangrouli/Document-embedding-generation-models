'''
Defines the method used to generate the AUROC score based on model outputs;
Defines the method used to plot the training / validation loss graph.
'''

from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.ticker as mticker
import torch

def plot_loss(history):
    '''
    Plot the trianing / validation loss of the model trianing process.
    
    @ history (dictionary): The dictionary that caches the training losses and validation losses through the epochs.
    '''
    x = range(1, len(history['train loss'])+1)
    y_train = history['train loss']
    y_val = history['val loss']

    plt.plot(x, y_train, label = 'training loss')
    plt.plot(x, y_val, label = 'validation loss')
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.xlabel('Number of Epoch')
    plt.ylabel('Loss')
    plt.title('Model Training Performance')
    plt.legend(loc = 'best')
    plt.show()

def ProduceAUC():
    ''' Generate the AUROC score and the ROC graph. '''
    
    score_df = torch.load("score.pt")
    print(score_df.columns)
    record = input('Enter the record name:')

    true_labels = torch.load('test_labels.pt')

    predictions_labels = list(score_df[record])

    fpr, tpr, _ = roc_curve(true_labels, predictions_labels)
    auc = roc_auc_score(true_labels, predictions_labels)
    print("AUROC:", auc)
    plt.plot(fpr,tpr,label="AUROC = "+str(auc))
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.title('ROC Curve of Model Prediction')
    plt.legend(loc=4)
    plt.show()
