# -*- coding: utf-8 -*-
import os
import gc
import argparse
import json
import random
import math
import random
from functools import reduce
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split, ShuffleSplit, StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support, classification_report
from sklearn import metrics
import torch
from torch import nn
from torch.optim import Adam, SGD, AdamW
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, CyclicLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from performer_pytorch import PerformerLM
import scanpy as sc
import anndata as ad
from utils import *
import pickle as pkl
import datetime

import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=-1, help='pred_class process rank.')
parser.add_argument("--bin_num", type=int, default=5, help='Number of bins.')
# parser.add_argument("--gene_num", type=int, default=16906, help='Number of genes.')
parser.add_argument("--epoch", type=int, default=100, help='Number of epochs.')
parser.add_argument("--seed", type=int, default=2021, help='Random seed.')
parser.add_argument("--batch_size", type=int, default=2, help='Number of batch size.')
parser.add_argument("--learning_rate", type=float, default=1e-4, help='Learning rate.')
parser.add_argument("--grad_acc", type=int, default=60, help='Number of gradient accumulation.')
parser.add_argument("--valid_every", type=int, default=1, help='Number of training epochs between twice validation.')
parser.add_argument("--pos_embed", type=bool, default=True, help='Using Gene2vec encoding or not.')
parser.add_argument("--cor_embed", type=bool, default=True, help='Using node2vec encoding or not.')
parser.add_argument("--data_name", type=str, default='YAN', help='Path of data for finetune.')
# parser.add_argument("--data_path", type=str, default='./dataset_path/Zeisel/train_preprocessed_data.h5ad', help='Path of data for finetune.')
# parser.add_argument("--valid_path", type=str, default='./dataset_path/Zeisel/test_preprocessed_data.h5ad', help='Path of data for finetune.')
parser.add_argument("--ckpt_dir", type=str, default='./datasets/', help='Directory of checkpoint to save.')
# parser.add_argument("--model_path", type=str, default='./file_required/panglao_pretrain.pth', help='Path of pretrained model.')

parser.add_argument("--model_name", type=str, default='Zeisel_finetune', help='Finetuned model name.')

args = parser.parse_known_args()[0]


data_train_ = sc.read_h5ad('./datasets/'+args.data_name+'/train_preprocessed_data.h5ad')
label_dict, label_train = np.unique(np.array(data_train_.obs['celltype']), return_inverse=True)  # Convert strings categorical to integrate categorical, and label_dict[label] can be restored
#store the label dict and label for prediction
with open('./datasets/'+args.data_name+'/label_dict', 'wb') as fp:
    pkl.dump(label_dict, fp)
with open('./datasets/'+args.data_name+'/label', 'wb') as fp:
    pkl.dump(label_train, fp)
class_num = np.unique(label_train, return_counts=True)[1].tolist()
class_weight = torch.tensor([(1 - (x / sum(class_num))) ** 2 for x in class_num])
label_train = torch.from_numpy(label_train)
data_train = data_train_.X

la= sc.read_h5ad('./datasets/'+args.data_name+'/test_preprocessed_data.h5ad')
data_val = sc.read_h5ad('./datasets/'+args.data_name+'/test_preprocessed_data.h5ad')
label_dict, label_val = np.unique(np.array(data_val.obs['celltype']), return_inverse=True)  # Convert strings categorical to integrate categorical, and label_dict[label] can be restored
#store the label dict and label for prediction
label_val = torch.from_numpy(label_val)
data_val = data_val.X


is_master = 1
SEED = args.seed
EPOCHS = args.epoch
BATCH_SIZE = args.batch_size
GRADIENT_ACCUMULATION = args.grad_acc
LEARNING_RATE = args.learning_rate
SEQ_LEN = len(data_train_.var) + 1
VALIDATE_EVERY = args.valid_every

PATIENCE = 10
UNASSIGN_THRES = 0.0
CLASS = args.bin_num + 2

POS_EMBED_USING = args.pos_embed
COR_EMBED_USING = args.cor_embed

model_name = args.model_name
ckpt_dir = args.ckpt_dir+args.data_name+'/'

# dist.init_process_group(backend='nccl')
# torch.cuda.set_device(local_rank)
device = torch.device("cuda")
# world_size = torch.distributed.get_world_size()

seed_all(SEED)


class SCDataset(Dataset):
    def __init__(self, data, label):
        super().__init__()
        self.data = data
        self.label = label

    def __getitem__(self, index):
        rand_start = random.randint(0, self.data.shape[0]-1)
        full_seq = self.data[rand_start].toarray()[0]
        full_seq[full_seq > (CLASS - 2)] = CLASS - 2
        full_seq = torch.from_numpy(full_seq).long()
        full_seq = torch.cat((full_seq, torch.tensor([0]))).to(device)
        seq_label = self.label[rand_start]
        # seq_label = self.label
        
        return full_seq, seq_label

    def __len__(self):
        return self.data.shape[0]


class Identity(torch.nn.Module):
    def __init__(self, dropout = 0., h_dim = 100, out_dim = 10):
        super(Identity, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, (1, 200))
        self.act = nn.ReLU()
        self.fc1 = nn.Linear(in_features=SEQ_LEN, out_features=512, bias=True)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(in_features=512, out_features=h_dim, bias=True)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(in_features=h_dim, out_features=out_dim, bias=True)

    def forward(self, x):
        x = x[:,None,:,:]
        x = self.conv1(x)
        x = self.act(x)
        x = x.view(x.shape[0],-1)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x



acc = []
f1 = []
f1w = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
pred_list = pd.Series(['un'] * data_train.shape[0])


train_dataset = SCDataset(data_train, label_train)
val_dataset = SCDataset(data_val, label_val)


train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

model = PerformerLM(
    num_tokens = CLASS,
    dim = 200,
    depth = 6,
    max_seq_len = SEQ_LEN,
    heads = 10,
    local_attn_heads = 0,
    g2v_position_emb = POS_EMBED_USING,
    n2v_position_emb = COR_EMBED_USING,
    path = './datasets/'+args.data_name
)

# path = args.model_path
# ckpt = torch.load(path)
# model.load_state_dict(ckpt['model_state_dict'])
# for param in model.parameters():
#     param.requires_grad = False
# for param in model.norm.parameters():
#     param.requires_grad = True
# for param in model.performer.net.layers[-2].parameters():
#     param.requires_grad = True

model.to_out = Identity(dropout=0., h_dim=128, out_dim=label_dict.shape[0])
max_acc = 0.0
file_path='./datasets/'+args.data_name+'/'+'best.pt'
folder = os.path.exists(file_path)
if folder:
    ckpt = torch.load(file_path)
    model.load_state_dict(ckpt['model_state_dict'])
    max_acc = ckpt['best_acc']
model = model.to(device)
# model = DDP(model, device_ids=[local_rank], output_device=local_rank)

# optimizer
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = CosineAnnealingWarmupRestarts(
    optimizer,
    first_cycle_steps=15,
    cycle_mult=2,
    max_lr=LEARNING_RATE,
    min_lr=1e-6,
    warmup_steps=5,
    gamma=0.9
)
loss_fn = nn.CrossEntropyLoss(weight=None)
# .to(local_rank)

# dist.barrier()
trigger_times = 0

flag_train=0
for i in range(1, EPOCHS+1):
    time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(time)
    # train_loader.sampler.set_epoch(i)
    model.train()
    # dist.barrier()
    running_loss = 0.0
    cum_acc = 0.0
    for index, (data, labels) in enumerate(train_loader):
        index += 1
        data, labels = data.to(device), labels.to(device)
        logits = model(data)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        running_loss += loss.item()
        
        softmax = nn.Softmax(dim=-1)
        final = softmax(logits)
        final = final.argmax(dim=-1)
        pred_num = labels.size(0)
        correct_num = torch.eq(final, labels).sum(dim=-1)
        cum_acc += torch.true_divide(correct_num, pred_num).mean().item()
    epoch_loss = running_loss / index
    epoch_acc = 100 * cum_acc / index
    # epoch_loss = get_reduced(epoch_loss, local_rank, 0, world_size)
    # epoch_acc = get_reduced(epoch_acc, local_rank, 0, world_size)
    if is_master:
        print(f'    ==  Epoch: {i} | Training Loss: {epoch_loss:.6f} | Accuracy: {epoch_acc:6.4f}%  ==')
    # dist.barrier()
    scheduler.step()


    model.eval()
    running_loss = 0.0
    batch_size = data_val.shape[0]
    pred_finals = []
    novel_indices = []
    pred_class=[]
    truth_finals=[]
    with torch.no_grad():
        for index in range(batch_size):
            full_seq = data_val[index].toarray()[0]
            full_seq[full_seq > (CLASS - 2)] = CLASS - 2
            full_seq = torch.from_numpy(full_seq).long()
            full_seq = torch.cat((full_seq, torch.tensor([0]))).to(device)
            full_seq = full_seq.unsqueeze(0)
            pred_logits = model(full_seq)
            # print(la.obs['celltype'][index])
            ground_truth = torch.Tensor([la.obs['celltype'][index]])
            # print(pred_logits,pred_logits.shape,ground_truth,ground_truth.shape)
            ground_truth = ground_truth.type(torch.LongTensor)
            loss = loss_fn(pred_logits.cpu(), ground_truth)
            running_loss += loss.item()        
            softmax = nn.Softmax(dim=-1)
            pred_prob = softmax(pred_logits)
            pred_class.append(pred_prob.cpu())
            pred_final = pred_prob.argmax(dim=-1)
            pred_finals.append(pred_final.cpu().tolist())
            truth_finals.append(ground_truth.cpu().tolist())
        pred_finals=sum(pred_finals,[])
        pred_list = label_dict[pred_finals].tolist()
        # print(classification_report(la.obs['celltype'], pred_list, target_names=None, digits=4))          
        f1 = f1_score(la.obs['celltype'], pred_list, average='macro')
        
       

        cur_acc = accuracy_score(truth_finals, pred_finals)
        val_loss = running_loss / batch_size               
#         val_loss = running_loss / index
        if is_master:
            print(f'    ==  Epoch: {i} | Validation Loss: {val_loss:.6f} | F1 Score: {f1:.6f}  ==')
            print('ACC score is ', cur_acc)
            print('ARI score is ', metrics.adjusted_rand_score(la.obs['celltype'], pred_list))
            print('NMI score is ', metrics.normalized_mutual_info_score(la.obs['celltype'], pred_list)) 
            # print(confusion_matrix(pred_finals, pred_list))
            # print(classification_report(truth_finals, predictions, target_names=label_dict.tolist(), digits=4))
            print(classification_report(la.obs['celltype'], pred_list, target_names=None, digits=4))                
            
        if cur_acc > max_acc:
            max_acc = cur_acc
            trigger_times = 0
            # save_best_ckpt(i, model, optimizer, scheduler, val_loss, model_name, ckpt_dir)
            torch.save({
                'model_state_dict': model.state_dict(),
                'best_acc':max_acc
                       },ckpt_dir+'best.pt')
        else:
            trigger_times += 1
            if trigger_times > PATIENCE:
                break
    # del predictions, 
    
ckpt = torch.load(file_path)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
# with open('./dataset_path/'+args.data_name+'/label_dict', 'rb') as fp:
#     label_dict = pkl.load(fp)

batch_size = data_val.shape[0]
batch_size2 = data_train.shape[0]
pred_finals = []
novel_indices = []
pred_class=[]

with torch.no_grad():
    for index in range(batch_size2):
        full_seq = data_train[index].toarray()[0]
        full_seq[full_seq > (CLASS - 2)] = CLASS - 2
        full_seq = torch.from_numpy(full_seq).long()
        full_seq = torch.cat((full_seq, torch.tensor([0]))).to(device)
        full_seq = full_seq.unsqueeze(0)
        pred_logits = model(full_seq)
        softmax = nn.Softmax(dim=-1)
        pred_prob = softmax(pred_logits)
        pred_class.append(pred_prob.cpu())
        pred_final = pred_prob.argmax(dim=-1)
        pred_finals.append(pred_final.cpu().tolist())

with torch.no_grad():
    for index in range(batch_size):
        full_seq = data_val[index].toarray()[0]
        full_seq[full_seq > (CLASS - 2)] = CLASS - 2
        full_seq = torch.from_numpy(full_seq).long()
        full_seq = torch.cat((full_seq, torch.tensor([0]))).to(device)
        full_seq = full_seq.unsqueeze(0)
        pred_logits = model(full_seq)
        softmax = nn.Softmax(dim=-1)
        pred_prob = softmax(pred_logits)
        pred_class.append(pred_prob.cpu())
        pred_final = pred_prob.argmax(dim=-1)
        pred_finals.append(pred_final.cpu().tolist())
pred_finals=sum(pred_finals,[])
pred_list = label_dict[pred_finals].tolist()


a=torch.cat(pred_class,dim=0)
print(a.shape)
x = a.cpu().numpy()
data_df = pd.DataFrame(x)
data_df.to_csv('./datasets/'+args.data_name+'/Gene_L.csv',index=False)
