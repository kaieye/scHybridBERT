import pandas as pd
import scanpy as sc
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import anndata as ad
from scipy import sparse
import os
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import networkx as nx
import matplotlib.pyplot as plt
from fastnode2vec import Node2Vec,Graph
from gensim.models import Word2Vec
import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--data_name", type=str, default='Zeisel', help='Local process rank.')
args = parser.parse_args()

ds_name = args.data_name
file_path='./datasets/'+ds_name+'/'
dataset_path = './datasets/'+ds_name+'/'
folder = os.path.exists(dataset_path)
if not folder:
    os.makedirs(dataset_path) 
    
label0= pd.read_csv(file_path+'label.csv')
class_name=list(set(label0.iloc[:,1]))
label=label0.iloc[:,1]


if type(label[0])!=str:
    tmp=list(set(label))
    if int(tmp[0])!=0:
        label-=1
else:
    lab={}
    for i in range(len(class_name)):
        lab.update({str(i):class_name[i]})
    for i in range(len(label)):
        for j in range(len(set(lab))):
            if lab[str(j)]==label[i]:
                label[i]=j
label=label.astype(int)
data = pd.read_csv(file_path+'data.csv', sep=',', encoding='utf-8')
# data=data.T.iloc[1:,:] # for cell row/columns
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=44)
for index_train, index_test in sss.split(data, label):
    data_train,label_train=data.iloc[index_train],label.iloc[index_train]
    data_test,label_test=data.iloc[index_test],label.iloc[index_test]
data_train.to_csv(dataset_path+'train_data.csv',index=False)
data_test.to_csv(dataset_path+'test_data.csv',index=False)
label_train.to_csv(dataset_path+'train_label.csv',index=False)
label_test.to_csv(dataset_path+'test_label.csv',index=False)

# 对于data.csv的格式，不同数据集有所差别，可能是每行或每列为一个细胞的基因表达，因此若行为一个细胞的基因表达，则需要转置
# 该部分生成数据供图边数据生成使用
data = pd.read_csv(dataset_path+'train_data.csv', sep=',', encoding='utf-8')
data2=data.iloc[:,1:].T
data2.columns=data.iloc[:,0]
data2.insert(loc=0, column='ID', value=list(data2.index))
data2=data2.reset_index(drop=True)

data_ = pd.read_csv(dataset_path+'test_data.csv', sep=',', encoding='utf-8')
data2_=data_.iloc[:,1:].T
data2_.columns=data_.iloc[:,0]
data2_.insert(loc=0, column='ID', value=list(data2_.index))
data2_=data2_.reset_index(drop=True)


data_concate = pd.concat([data2,data2_.iloc[:,1:]],axis=1)
data_concate.to_csv(dataset_path+'concate_ori_data.csv',index=False)





df=pd.read_csv(dataset_path+'concate_ori_data.csv',index_col=0)
df1 = df[df.astype('bool').mean(axis=1) >= 0.01]
# df2=df1
# criteriaGene = df1.astype('bool').mean(axis=0) >= 0.01
# df2 = df1[df1.columns[criteriaGene]]



# df2=df1
# criteriaSelectGene_seq = df2.var(axis=1).sort_values()[-int(0.25*len(df2)):]
# criteriaSelectGene_node = df2.var(axis=1).sort_values()[-int(0.25*len(df2)):]
# df3_seq = df2.loc[criteriaSelectGene_seq.index]
# df3_node = df2.loc[criteriaSelectGene_node.index]
df3_seq = df1
df3_node = df1



# if transform == 'log':
df3_seq = df3_seq.transform(lambda x: np.log(x + 1))
df3_node = df3_node.transform(lambda x: np.log(x + 1))
# df3_seq.to_csv(dataset_path+'concat_Use_expression.csv')
df3_node.to_csv(dataset_path+'concat_Use_expression.csv')
df3_seq.iloc[:,:int(df3_seq.shape[1]*0.8)].to_csv(dataset_path+'train_data.csv')
df3_seq.iloc[:,int(df3_seq.shape[1]*0.8):].to_csv(dataset_path+'test_data.csv')
os.remove(dataset_path+'concate_ori_data.csv')

ds=['train','test']
for i in ds:
    data = pd.read_csv(dataset_path+i+'_label.csv')
    data['id'] = [i for i in range(len(data))]
    data = data.iloc[:,[1,0]]
    data.to_csv(dataset_path+i+'_label.csv',index=False)
    data.columns = ['graph_id','label']
    data.to_csv(dataset_path+i+'_graph_properties.csv',index=False)

    data = pd.read_csv(dataset_path+i+'_label.csv')
    for j in range(len(data)):
        data.iloc[j,1] =str(data.iloc[j,1])
    data.to_csv(dataset_path+i+'_label.csv',index=False)

    
ds=['train','test']
# 以下部分供scbert使用
for i in ds:
    data=pd.read_csv(dataset_path+i+'_data.csv').T
    a=data.iloc[0,:]
    a=[str(i)+ '_' for i in a]
    data=data.iloc[1:,:]
    data.columns=a
    # data=data.rename(columns={'Unnamed: 0':'cell_id'})
    data.to_csv(dataset_path+i+'_data_fake_col.csv',index=False)

    a =  sc.read_csv(dataset_path+i+'_data_fake_col.csv')
    b =  pd.read_csv(dataset_path+i+'_data_fake_col.csv')
    ty = pd.read_csv(dataset_path+i+'_label.csv')
    ty['n_genes'] = list((b!= 0).sum(axis=1))
    a.obs['celltype'] = list(ty.iloc[:,1])
    a.obs['n_genes'] = list(ty['n_genes'])
    data = a

    counts = sparse.lil_matrix((data.X.shape[0],data.X.shape[1]),dtype=np.float32)
    # ref = panglao.var_names.tolist()
    obj = data.var_names.tolist()
    # print(len(ref))
    for s in range(len(obj)):
        loc = obj.index(obj[s])
        counts[:,s] = data.X[:,loc]
    counts = counts.tocsr()
    new = ad.AnnData(X=counts)
    new.var_names = obj
    new.obs_names = data.obs_names
    new.obs = data.obs
    # new.uns = panglao.uns
    # sc.pp.filter_cells(new, min_genes=150)
    sc.pp.normalize_total(new, target_sum=1e4)
    sc.pp.log1p(new, base=2)
    new.write(dataset_path+i+'_preprocessed_data.h5ad')
    
os.remove(dataset_path+'train_data_fake_col.csv')
os.remove(dataset_path+'test_data_fake_col.csv')
# os.remove(dataset_path+'test_data.csv')
# os.remove(dataset_path+'train_data.csv')
# os.remove(dataset_path+'test_label.csv')
# os.remove(dataset_path+'train_label.csv')

exp=pd.read_csv(dataset_path+'concat_Use_expression.csv')
h1=sc.read_h5ad(dataset_path+'test_preprocessed_data.h5ad')
h2=sc.read_h5ad(dataset_path+'train_preprocessed_data.h5ad')
print(exp.shape[1]==1+h1.n_obs+h2.n_obs)

# 读取基因表达矩阵
expression = pd.read_csv(dataset_path+'train_data.csv')
expression = expression.iloc[:,1:]
idx=expression.index

# 对表达矩阵进行标准化处理
scaler = StandardScaler()
expression = scaler.fit_transform(expression)

# 使用PCA降维
pca = PCA(n_components=50)
expression_pca = pca.fit_transform(expression)

# 计算基因间的相似性
similarity = pdist(expression_pca, metric='cosine')
similarity_matrix = squareform(similarity)

# 对相似性矩阵进行聚类分析
# linkage_matrix = linkage(similarity, method='ward')
# dendrogram(linkage_matrix, labels=idx)
# plt.show()

# 构建基因共表达网络
network = nx.Graph()
print(len(expression))
tmp=[]
time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print('-----start build gene network-----',time)
for i in range(len(expression)):
    if i%2000==0:
        print(i)
    flag=0
    for j in range(i+1, len(expression)):
        weight = 1 - similarity_matrix[i, j]
        if weight > 0.5: # 根据阈值确定是否连接边
            flag=1
            # network.add_edge(idx[i], idx[j], weight=weight)
            network.add_edge(i, j, weight=weight)
            tmp.append((i,j,weight))
            
    if(flag==0):
        # network.add_node(idx[i])
        network.add_node(i)
        tmp.append((i,i,0))
time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print('-----finished -----',time) 
# 可视化基因共表达网络
# pos = nx.spring_layout(network)


weights = [network[u][v]['weight'] for u, v in network.edges()]
# Generate node embeddings using node2vec


# node2vec = Node2Vec(network, dimensions=200, walk_length=10, num_walks=100)
# model = node2vec.fit(window=5, min_count=1, batch_words=4)
# emb = {str(node): model.wv[str(node)] for node in network.nodes()}
time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print('-----start gene-regulatory-embedding -----',time) 
graph = Graph(tmp,directed=False, weighted=True)
node2vec = Node2Vec(graph, dim=200, walk_length=100, window=10, p=2.0, q=0.5, workers=2)
node2vec.train(epochs=100)
emb = {str(node): node2vec.wv[node] for node in graph.node_names}
time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print('-----finished -----',time) 

res = np.array([list(item) for item in emb.values()])
np.save(dataset_path+'gene_interaction_graph.npy', res)


time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print('-----start gene-embedding -----',time) 
# 读取基因表达矩阵文件，假设每一行是一个基因，每一列是一个样本
df = pd.read_csv(dataset_path+"train_data.csv")
gene_names = [str(gene) for gene in range(df.shape[0])]
model = Word2Vec([gene_names], vector_size=200, window=10, min_count=0, workers=4)
np.save(dataset_path+'gene2vec.npy',model.wv.vectors)
time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print('-----finished -----',time) 

node2vec_weight = np.load(dataset_path+'gene_interaction_graph.npy')
gene2vec_weight = np.load(dataset_path+'gene2vec.npy')
print(node2vec_weight.shape==gene2vec_weight.shape,gene2vec_weight.shape)
