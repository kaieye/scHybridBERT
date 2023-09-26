import os
import sys
import time
import torch
import datetime
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import pickle as pkl
from torch import nn
import networkx as nx
from scipy import sparse
import scipy.sparse as sp
from sklearn import metrics
from scipy.spatial import distance
from torch.nn import functional as F
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import  StratifiedShuffleSplit

import dgl
from dgl import DGLGraph
from dgl.nn.pytorch.conv import SAGEConv
from dgl.nn.pytorch import GraphConv
from dgl.data import register_data_args, DGLDataset
from dgl.nn.pytorch import GraphConv,GATConv
# from dgl.nn.pytorch import WeightedGraphConv
from dgl.nn import GATv2Conv
parser = argparse.ArgumentParser(description='GraphSAGE')
register_data_args(parser)
parser.add_argument("--dropout", type=float, default=0.5,
                    help="dropout probability")
parser.add_argument("--gpu", type=int, default=1,
                    help="gpu")
parser.add_argument("--lr", type=float, default=1e-2,
                    help="learning rate")
parser.add_argument("--n-epochs", type=int, default=2000,
                    help="number of training epochs")
parser.add_argument("--n-hidden", type=int, default=128,#16
                    help="number of hidden gcn units")
parser.add_argument("--n-layers", type=int, default=2,
                    help="number of hidden gcn layers")
parser.add_argument("--weight-decay", type=float, default=5e-4,
                    help="Weight for L2 loss")
parser.add_argument("--aggregator-type", type=str, default="gcn",
                    help="Aggregator type: mean/gcn/pool/lstm")
parser.add_argument("--data_name", type=str, default="YAN",
                    help="No")
parser.add_argument("--dis_method", type=int, default=3,
                    help="1.eudlid 2.cor+Manhattan 3.cor*e^(alphaMan) 4.todo")
args = parser.parse_known_args()[0]

ds_name = args.data_name
dataset_path = './datasets/'+ds_name+'/'
folder = os.path.exists(dataset_path)
if not folder:
    os.makedirs(dataset_path) 
class AE(nn.Module):
    ''' Autoencoder for dimensional reduction'''
    def __init__(self,dim):
        super(AE, self).__init__()
        self.dim = dim
        self.fc1 = nn.Linear(dim, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 1024)
        self.fc4 = nn.Linear(1024, dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return F.relu(self.fc2(h1))

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.relu(self.fc4(h3))

    def forward(self, x):
        z = self.encode(x.view(-1, self.dim))
        return self.decode(z), z
    

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc21 = nn.Linear(256, latent_dim)
        self.fc22 = nn.Linear(256, latent_dim)
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim, 256)
        self.fc4 = nn.Linear(256, input_dim)
    
    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))
    
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar,z
CFG={
    'batch_size':12800,
    # 'datasetname':dataset[8:-1],
    'Regu_epochs':4000,
    'k':10
}


def train(epoch, train_loader):

    model_AE.train()
    train_loss = 0
    for batch_idx, (data, dataindex) in enumerate(train_loader):
        data = data.type(torch.FloatTensor)
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch,  mu, logvar,z = model_AE(data)
        
        mse_loss = nn.MSELoss(reduction='sum')
        kl_loss = nn.KLDivLoss(reduction='batchmean') # KL散度损失函数
        
        rec_loss = mse_loss(recon_batch, data)
        mu = torch.mean(z, dim=0)
        logvar = torch.log(torch.var(z, dim=0) + 1e-10)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        loss = 0.7*rec_loss + 0.3*kl
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        

        # for batch
        if batch_idx == 0:
            recon_batch_all = recon_batch
            data_all = data
            z_all = z
        else:
            recon_batch_all = torch.cat((recon_batch_all, recon_batch), 0)
            data_all = torch.cat((data_all, data), 0)
            z_all = torch.cat((z_all, z), 0)
    # if epoch % 200==0 or epoch==1 :
    #     print('====> Epoch: {} Average loss: {:.4f}'.format(
    #       epoch, train_loss / len(train_loader.dataset)))

    return z_all, train_loss / len(train_loader.dataset)


class scDataset(Dataset):
    def __init__(self, data=None, transform=None):
        """
        Args:
            data : sparse matrix.
            transform (callable, optional):
        """
        # Now lines are cells, and cols are genes
        self.features = data.transpose()

        # save nonzero
        # self.nz_i,self.nz_j = self.features.nonzero()
        self.transform = transform

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.features[idx, :]
        if type(sample) == sp.lil_matrix:
            sample = torch.from_numpy(sample.toarray())
        else:
            sample = torch.from_numpy(sample)

        # transform after get the data
        if self.transform:
            sample = self.transform(sample)

        return sample, idx
    
data = pd.read_csv(dataset_path+'concat_Use_expression.csv', index_col=0)
data = data.to_numpy()
data = data.astype(float)
scData = scDataset(data)
regulationMatrix = None
# train_loader = DataLoader(scData, batch_size=CFG['batch_size'], shuffle=False, **kwargs)
train_loader = DataLoader(scData, batch_size=CFG['batch_size'], shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_AE = VAE(scData.features.shape[1],128).to(device)

file_path='./datasets/'+args.data_name+'/VAE.pt'
folder = os.path.exists(file_path)

if folder:
    ckpt = torch.load(file_path)
    model_AE.load_state_dict(ckpt['model_state_dict'])
    # max_acc = ckpt['best_acc']
model_AE = model_AE.to(device)



optimizer = Adam(model_AE.parameters(), lr=1e-3)

start_time = time.time()

print('Start training...')
pre_200_loss=1e10
# folder=0
if folder:
    z, tmp_loss= train(1, train_loader)
else:
    for epoch in range(1, CFG['Regu_epochs'] + 1):
        z, tmp_loss= train(epoch, train_loader)
        if epoch%200==0:
            print("tmp_loss is ",tmp_loss)
            # if pre_200_loss-tmp_loss<=1:
            #     break
            # pre_200_loss=tmp_loss
            torch.save({'model_state_dict':model_AE.state_dict()},'./datasets/'+args.data_name+'/'+'VAE.pt')
zOut = z.detach().cpu().numpy()
print('zOut ready at ' + str(time.time()-start_time))

def generateAdj(featureMatrix, k = 10, method=None):
    """
    Generating edgeList 
    """
    edgeList=[]
    # Version 1: cost memory, precalculate all dist
    if method==1:
        distMat = distance.cdist(featureMatrix,featureMatrix, 'euclidean')
    elif method==2:
        distMat1 = distance.cdist(featureMatrix,featureMatrix, 'correlation')
        distMat2 = distance.cdist(featureMatrix,featureMatrix, 'cityblock')
        distMat = 0.7*distMat1+0.3*distMat2
        # distMat = distance.cdist(featureMatrix,featureMatrix, 'pearson')
    elif method==3:
        distMat1 = distance.cdist(featureMatrix,featureMatrix, 'correlation')
        distMat2 = distance.cdist(featureMatrix,featureMatrix, 'cityblock')
        distMat = distMat1 * np.exp(-0.25 * distMat2)
    elif method==4:
        distMat1 = distance.cdist(featureMatrix,featureMatrix, 'euclidean')
        distMat2 = distance.cdist(featureMatrix,featureMatrix, 'correlation')
        distMat = 0.7*distMat1+0.3*distMat2
    # print(distMat.shape)
    # parallel
    # distMat = pairwise_distances(featureMatrix,featureMatrix, distanceType, n_jobs=-1)
    
    for i in np.arange(distMat.shape[0]):
        res = distMat[:,i].argsort()[:k+1]
        tmpdist = distMat[res[1:k+1],i]
        mean = np.mean(tmpdist)
        std = np.std(tmpdist)
        flag=0
        for j in np.arange(1,k+1):
            
            if (distMat[i,res[j]]<=mean+std) and (distMat[i,res[j]]>=mean-std):
            # if(distMat[i,res[j]])
                weight = 1.0
                edgeList.append((i,res[j],weight))
                flag=1
            else:
                weight = 0.0
            # edgeList.append((i,res[j],weight))
        if flag==0:
            edgeList.append((i,i,1))

    return edgeList



print('Use ',args.dis_method,' method')
edgeList = generateAdj(zOut,  k=CFG['k'] ,method=args.dis_method)

graph_df = pd.DataFrame(edgeList, columns=["Src", "Dst", "Weight"])
graph_df.to_csv(dataset_path+'concat_graph_edges.csv', index=False)

class GATSAGE(nn.Module):
    def __init__(self,in_feats,n_hidden,n_classes,n_layers,activation,dropout,aggregator_type,n_heads=8):
        super(GATSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.n_layers = n_layers
        # input layer
        self.layer1=SAGEConv(in_feats, n_hidden, aggregator_type)
        # hidden layers
        self.layer2=SAGEConv(n_hidden, n_hidden, aggregator_type)
        # output layer
        # self.layer3=SAGEConv(n_hidden, n_classes, aggregator_type) # activation None
        in_feats = n_hidden
        
        self.gat_convs = nn.ModuleList()
        for i in range(1):
            self.gat_convs.append(GATConv(in_feats, n_hidden, n_heads)) 
            in_feats = n_hidden * n_heads
        self.classify = nn.Linear(in_feats, n_classes)
    def forward(self, graph, inputs):
        recon_batch,  mu, logvar,z = model_AE(inputs)
        h = self.dropout(inputs)
        h = self.layer1(graph, h)
        h+=z
        h = self.activation(h)
        
        h = self.dropout(h)
        h = self.layer2(graph, h)
        h+=z
        h = self.activation(h)
      
        # h = self.dropout(h)
        # h = self.layer3(graph, h)
        # return h
        
        for i in range(1):
            h = self.gat_convs[i](g, h).flatten(1)
        logits = self.classify(h) 
        return logits

def evaluate(model, graph, features, labels, nid):
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)
        logits = logits[nid]
        labels = labels[nid]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)
class scRNA(DGLDataset):
    def __init__(self,num_class):
        super().__init__('wcj')
        self.num_classes=num_class
        self.num_node_features=num_class
    def process(self):
        train_nodes_data = pd.read_csv(dataset_path+'train_graph_properties.csv')
        test_nodes_data = pd.read_csv(dataset_path+'test_graph_properties.csv')
        nodes_data = pd.concat([train_nodes_data,test_nodes_data])
        edges_data=pd.read_csv(dataset_path+'concat_graph_edges.csv')
        fea_data = pd.read_csv(dataset_path+'concat_Use_expression.csv')

        
        fea_data = fea_data.iloc[:,1:].values.T
        node_features = torch.from_numpy(np.array(fea_data))
        node_features = node_features.type(torch.FloatTensor)
        node_labels = torch.from_numpy(nodes_data['label'].to_numpy())
        edge_features = torch.from_numpy(edges_data['Weight'].to_numpy())
        edges_src = torch.from_numpy(edges_data['Src'].to_numpy())
        edges_dst = torch.from_numpy(edges_data['Dst'].to_numpy())

        self.graph = dgl.graph((edges_src, edges_dst), num_nodes=nodes_data.shape[0])
        self.graph.ndata['feat'] = node_features
        self.graph.ndata['label'] = node_labels
        self.graph.edata['weight'] = edge_features

        n_nodes = nodes_data.shape[0]
        n_train = int(n_nodes * 0.8)
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[:n_train] = True
        
        val_mask[n_train:] = True
        # test_mask[n_train + n_val:] = True
        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        # self.graph.ndata['test_mask'] = test_mask

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1
nodes_data = pd.read_csv(dataset_path+'train_graph_properties.csv')
classes=len(set(nodes_data.iloc[:,1]))

data = scRNA(classes)
g = data[0]

features = g.ndata['feat']
labels = g.ndata['label']
train_mask = g.ndata['train_mask']
val_mask = g.ndata['val_mask']
# test_mask = g.ndata['test_mask']
in_feats = features.shape[1]
n_classes = data.num_classes
n_edges = g.number_of_edges()
print("""----Data statistics------'
  #Edges %d
  #Classes %d
  #Train samples %d
  #Val samples %d""" %
      (n_edges, n_classes,
       train_mask.int().sum().item(),
       val_mask.int().sum().item()))
args.gpu=0
if args.gpu < 0:
    cuda = False
else:
    cuda = True
    torch.cuda.set_device(args.gpu)
    features = features.cuda()
    labels = labels.cuda()
    train_mask = train_mask.cuda()
    val_mask = val_mask.cuda()
    # test_mask = test_mask.cuda()
    print("use cuda:", args.gpu)

train_nid = train_mask.nonzero().squeeze()
val_nid = val_mask.nonzero().squeeze()
# test_nid = test_mask.nonzero().squeeze()

# graph preprocess and calculate normalization factor
g = dgl.remove_self_loop(g)
n_edges = g.number_of_edges()
if cuda:
    g = g.int().to(args.gpu)
heads = ([16] * args.n_layers) + [1]
# model = GATv2(in_feats,args.n_hidden,n_classes,args.n_layers,heads,F.relu,args.dropout,0.0001,0.2,False)
# model = GraphSAGEGAT(in_feats,args.n_hidden,n_classes,3,8)
model = GATSAGE(in_feats,args.n_hidden,n_classes,args.n_layers,F.relu,args.dropout,args.aggregator_type)
if cuda:
    model.cuda()
g = dgl.add_self_loop(g)
# use optimizer
optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# initialize graph
dur = []
best_acc=0
patience=200
time_cur = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print('start model training',time_cur)
for epoch in range(0,args.n_epochs+1):
    model.train()
    if epoch >= 3:
        t0 = time.time()
    logits = model(g, features)

    loss = F.cross_entropy(logits[train_nid], labels[train_nid])
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch >= 3:
        dur.append(time.time() - t0)

    acc = evaluate(model, g, features, labels, val_nid)

    if epoch%200==0:
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
          "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(),
                                        acc, n_edges / np.mean(dur) / 1000))

    if acc>best_acc:
        torch.save(model.state_dict(), dataset_path+'graphsage.pth')
        best_acc=acc
        trigger_times = 0
    else:
        trigger_times +=1
        # if trigger_times >patience:
        #     print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
        #   "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(),
        #                                 acc, n_edges / np.mean(dur) / 1000))
        #     break
time_cur = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print('finished',time_cur)           
model.load_state_dict(torch.load(dataset_path+'graphsage.pth'))
model.eval()
with torch.no_grad():
    logits = model(g, features)
    logits = logits[val_nid]
x=logits.cpu().numpy()
data_df = pd.DataFrame(x)
data_df.to_csv(dataset_path+'Cell_L.csv',index=False)
