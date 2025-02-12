from __future__ import division
from __future__ import print_function
from dgl.data import WisconsinDataset
import time
import numpy as np
import torch



def adjacency_matrix_to_adjacency_list_with_weights(adj_matrix):
    n, m = len(adj_matrix), len(adj_matrix[0])
    u_adj_dict = {}  # u 
    v_adj_dict = {}  # v 

    for i in range(n):
        u_adj_dict[i] = []  # init u 
        for j in range(m):
            if adj_matrix[i][j] != 0:
                u_adj_dict[i].append(j+n)  #
                if j+n not in v_adj_dict:
                    v_adj_dict[j+n] = []  # 
                v_adj_dict[j+n].append(i)  # 
        for j in range(m):
            if j+n not in v_adj_dict:
                    v_adj_dict[j+n] = []

    return u_adj_dict, v_adj_dict

def check_feat(feat):
        A = feat.T
        for i in range(A.shape[0]):

            for j in range((A.shape[1])):
                     if A[i][j]!=0:
                         break
                     if j == (A.shape[1])-1 and A[i][j]==0:
                         print('There is all 0 feature ')
                         exit()

def get_degree_list(adj_list,flag,n):
    # print(u_adj_list)
    # exit()
    if flag=='u':
        degree_list={}
        for i in range(len(adj_list)):
            degree_list[i]=len(adj_list[i])
            # degree_list.append(len(u_adj_list[i]))
    else:
        degree_list={}
        for i in range(len(adj_list)):
            adj_list[i+n]
            degree_list[i+n]=len(adj_list[i+n])
            # degree_list.append(len(u_adj_list[i+n]))   
        # degree_list.append(len(v))
    
    return degree_list

def fpush(seed,alpha,epsilon,u_list,v_list,u_deg, v_deg):  #NIBBLE_PPR
    
        starttime = time.time()
        pi, r = {}, {}
        li=[]
        r[seed] = 1
        q = collections.deque()
        q.append(seed)
        while q:
            u = q.popleft()
            for v in u_list[u]:
                if v not in r:
                    r[v] = 0
                update=(1 - alpha) * r[u] / u_deg[u]
                r[v] = r[v] + update  # unweighted graph
            if u not in pi:
                li.append(u)
                pi[u] = 0
            pi[u] = pi[u] + alpha * r[u]
            r[u] = 0

            for v in u_list[u]:
                for u_ in v_list[v]:
                    if u_ not in r:
                        r[u_] = 0
                    update = r[v] / v_deg[v]
                    r[u_] = r[u_] + update
                    if (r[u_]- update)/u_deg[u_]<epsilon and r[u_] / u_deg[u_] >= epsilon:
                        q.append(u_)
                r[v]=0



        endtime = time.time()
        # print(li)
        return pi,endtime-starttime




dataset = dgl.data.CoraGraphDataset()
g = dataset[0]
num_classes = dataset.num_classes
features = g.ndata["feat"]
train_mask = g.ndata["train_mask"]
jk = torch.where(torch.Tensor(ac['train_mask'])==True)[0]
val_mask = g.ndata["val_mask"]
test_mask = g.ndata["test_mask"]
label = g.ndata['label']

array = np.array([[1,0,1,0,0],
                  [1,1,0,0,1],
                  [1,0,0,0,1],
                  [1,0,0,1,0]])

tensor_array_a = torch.Tensor(array)
u,v = adjacency_matrix_to_adjacency_list_with_weights(array)


shape=(len(u),len(u))
# print
feature_graph=[]

#init feature_graph set numbers
S=3
for _ in range(S):
    feature_graph.append(torch.zeros(shape))

print(len(u))
print(len(v))

u_degree= get_degree_list(u,flag='u',n=len(u))
v_degree= get_degree_list(v,flag='v',n=len(u))

stime =time.time()
for i in range(len(u)):
    ppr,pprtime = fpush(seed=i,alpha=0.1,epsilon=1e-3,u_list=u,v_list=v,u_deg=u_degree,v_deg=v_degree)
    print(pprtime)
    sorted_ppr = sorted(ppr.items(), key=lambda x: x[1],reverse=True)
    top_k_keys = [item[0] for item in sorted_ppr]

    #construct knn，k=3，4，5
    for k in range(3):
        feature_graph[0][i][[top_k_keys[k]]]=1
    for k in range(4):
        feature_graph[1][i][[top_k_keys[k]]]=1
    for k in range(5):
        feature_graph[2][i][[top_k_keys[k]]]=1


torch.save(feature_graph[0], 'feature_graph_3.pt')
torch.save(feature_graph[1], 'feature_graph_4.pt')
torch.save(feature_graph[2], 'feature_graph_5.pt')
