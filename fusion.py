import argparse
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split, ShuffleSplit, StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support, classification_report
from sklearn import metrics
import torch
from torch import nn
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
parser = argparse.ArgumentParser()
parser.add_argument("--data_name", type=str, default='Mouse', help='data name.')
args = parser.parse_known_args()[0]
import pandas as pd
softmax = nn.Softmax(dim=-1)
a=torch.Tensor(pd.read_csv('./datasets/'+args.data_name+'/Cell_L.csv').values)
b=torch.Tensor(pd.read_csv('./datasets/'+args.data_name+'/Gene_L.csv').values)
merged_data = torch.cat((a,b),dim=1)
train_data=merged_data[:int(merged_data.shape[0]*0.8),:]
test_data=merged_data[int(merged_data.shape[0]*0.8):,:]
train_label = pd.read_csv('./datasets/'+args.data_name+'/train_label.csv').iloc[:,1]
train_label = torch.Tensor(train_label)
test_label = pd.read_csv('./datasets/'+args.data_name+'/test_label.csv').iloc[:,1]
test_label = torch.Tensor(test_label)

a_data=a[:int(a.shape[0]*0.8),:]
a= softmax(a)
_,idx=torch.max(a_data, dim=1)

print('Cell_L: ',torch.sum(idx==train_label)/train_data.shape[0])
accuracy_model1 = torch.sum(idx==train_label)/train_data.shape[0]
b_data=b[:int(b.shape[0]*0.8),:]
_,idx=torch.max(b_data, dim=1)
print('Gene_L: ',torch.sum(idx==train_label)/train_data.shape[0])
accuracy_model2 = torch.sum(idx==train_label)/train_data.shape[0]
weight_model1 = accuracy_model1 / (accuracy_model1 + accuracy_model2)
weight_model2 = accuracy_model2 / (accuracy_model1 + accuracy_model2)
merged_data = torch.cat((weight_model1*a,weight_model2*b),dim=1)
# merged_data = (weight_model1*a+weight_model2*b)
from sklearn.ensemble import GradientBoostingClassifier
train_data=merged_data[:int(merged_data.shape[0]*0.8),:]
test_data=merged_data[int(merged_data.shape[0]*0.8):,:]
clf = MLPClassifier().fit(train_data,train_label)
pred = clf.predict(test_data)
print('scHybridBERT ACC :',round(int(torch.sum(torch.tensor(pred)==test_label))/test_data.shape[0],6))
print('ARI score is ',metrics.adjusted_rand_score(pred,test_label))
print('NMI score is ',metrics.normalized_mutual_info_score(pred,test_label))
