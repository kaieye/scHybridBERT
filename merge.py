import argparse
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split, ShuffleSplit, StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support, classification_report
from sklearn import metrics
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--data_name", type=str, default='Pbmc', help='data name.')
args = parser.parse_known_args()[0]


a=torch.Tensor(pd.read_csv('./datasets/'+args.data_name+'/SAGE.csv').values)
b=torch.Tensor(pd.read_csv('./datasets/'+args.data_name+'/scbert.csv').values)

merged_data = torch.cat((a,b),dim=1)
train_data=merged_data[:int(merged_data.shape[0]*0.8),:]
test_data=merged_data[int(merged_data.shape[0]*0.8):,:]

train_label = pd.read_csv('./datasets/'+args.data_name+'/train_label.csv').iloc[:,1]
train_label = torch.Tensor(train_label)
test_label = pd.read_csv('./datasets/'+args.data_name+'/test_label.csv').iloc[:,1]
test_label = torch.Tensor(test_label)


a_data=a[int(a.shape[0]*0.8):,:]
_,idx=torch.max(a_data, dim=1)
# print('SAGE: ',torch.sum(idx==test_label)/test_data.shape[0])

b_data=b[int(b.shape[0]*0.8):,:]
_,idx=torch.max(b_data, dim=1)
# print('bert: ',torch.sum(idx==test_label)/test_data.shape[0])


from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
clf = MLPClassifier(max_iter=10000).fit(train_data,train_label)
y=clf.predict(test_data)
print("HybridBERT")
print('ARI score is ',metrics.adjusted_rand_score(y, test_label))
print('NMI score is ',metrics.normalized_mutual_info_score(y, test_label))
print('ACC score is ',clf.score(test_data, test_label))
