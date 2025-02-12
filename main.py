
from scipy.sparse import csc_matrix, coo_matrix,lil_matrix
import arguments
import time
from utils import *

from early_stop import EarlyStopping, Stop_args
from load_data import load_data

from models import VGAE,GCN
import torch.nn as nn
import torch
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
import torch_geometric.utils as pygutiles
args = arguments.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

best_acc_val=0
def best_A_aug(A, acc_val=0, training=True):
    if acc_val>best_acc_val and training==True:
        best_acc_val=acc_val

    acc_val
    pass
def get_knn_feature_graph():
        feature_graph_list =[]
        for i in range(4):
            if i==0:
                f_matrix = torch.load('feature_graph_5.pt')
                f_matrix = f_matrix+f_matrix.t()
                f_matrix = f_matrix + torch.eye(f_matrix.shape[0])
                feature_graph_list.append(f_matrix)
            if i==1:
                f_matrix = torch.load('feature_graph_6.pt')
                f_matrix = f_matrix+f_matrix.t()
                f_matrix = f_matrix + torch.eye(f_matrix.shape[0])
                feature_graph_list.append(f_matrix)
            if i==2:
                f_matrix = torch.load('feature_graph_4.pt')
                f_matrix = f_matrix+f_matrix.t()
                f_matrix = f_matrix + torch.eye(f_matrix.shape[0])
                feature_graph_list.append(f_matrix)
            if i==3:
                f_matrix = torch.load('feature_graph_7.pt')
                f_matrix = f_matrix+f_matrix.t()
                f_matrix = f_matrix + torch.eye(f_matrix.shape[0])
                
                feature_graph_list.append(f_matrix)
        return feature_graph_list
def main():

    data, meta = load_data(args.dataset, args.train_per_class, args.val_per_class,
                           args.missing_link, args.missing_feature,
                           normalize_features=args.normalize_features, ogb_train_ratio=args.ogb_train_ratio,use_public_split=True)

    x = data.x.to(device)
    labels = data.y.to(device)
  
    idx_train = torch.where(data.train_mask==True)[0]  
  
    idx_val = torch.where(data.val_mask == True)[0]
    idx_test = torch.where(data.test_mask == True)[0]
    num_nodes = data.num_nodes
    num_classes = meta['num_classes']

    A = pygutiles.to_torch_coo_tensor(data.edge_index)
    
    A_selfloop = pygutiles.add_self_loops(data.edge_index)[0]
    A_selfloop= pygutiles.to_torch_coo_tensor(A_selfloop)
    
    normaliz_dad = normalize_adj_tensor(A_selfloop).to(device)
    
    
   
    model = GCN(nfeat=x.shape[1],nhid=128,nclass=num_classes,dropout=0.5).to(device)
   
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    feature_graph_list = get_knn_feature_graph()
    for i in range(4):
        feature_graph_list[i] = normalize_adj_tensor(feature_graph_list[i].to_sparse_coo()).to(device)


    stopping_args = Stop_args(patience=args.patience, max_epochs=args.epochs)
    early_stopping = EarlyStopping(model, **stopping_args)

    def test(labels):
        model.eval()
     
        _, output = model(x=x,adj=normaliz_dad)

        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        test_acc = accuracy(output[idx_test], labels[idx_test])

        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(test_acc.item()),
              )
        return test_acc.item()
    
    def consis_loss(logps, temp=0.6):
        ps = [torch.exp(p) for p in logps]
        sum_p = 0.
        for p in ps:
            sum_p = sum_p + p
        avg_p = sum_p/len(ps)
        
        
        sharp_p = (torch.pow(avg_p, 1./temp) / torch.sum(torch.pow(avg_p, 1./temp), dim=1, keepdim=True)).detach()
        loss = 0.
        for p in ps:
            loss += torch.mean((p-sharp_p).pow(2).sum(1))
        loss = loss/len(ps)
        return 1 * loss

  

    def train_full_batch():
        model.train()
        model.zero_grad()
        
        feat, output = model(x=x,adj=normaliz_dad)
        
        ce_loss = F.nll_loss(output[idx_train], labels[idx_train])
        loss_train = ce_loss
        X_list = []
       
        if args.lambda_ce_aug > 0:
            for i in range(4):
                
                feat_aug, output_aug = model(x=x,adj=feature_graph_list[i])
                X_list.append(output_aug)  
                ce_loss_aug = F.nll_loss(output_aug[idx_train], labels[idx_train])
                loss_train += ce_loss_aug * args.lambda_ce_aug

        loss_consis = consis_loss(X_list)

        if args.lambda_pa > 0:
            output_exp = torch.exp(output)
            confidences = output_exp.max(1)[0]
            pseudo_labels = output_exp.max(1)[1].type_as(labels)
            pseudo_labels[idx_train] = labels[idx_train]
            confidences[idx_train] = 1.0

            proto_aug = get_proto_norm_weighted(num_classes, feat_aug, pseudo_labels, confidences)
            proto = get_proto_norm_weighted(num_classes, feat, pseudo_labels, confidences)

            loss_pa = proto_align_loss(proto_aug, proto)
            loss_train += loss_pa * args.lambda_pa
        loss_train += loss_consis*args.lambda_consis
        
        loss_train.backward()
        optimizer.step()
      
    def train_mini_batch(batch_size):
        idx_not_train = torch.where(data.train_mask == False)[0]
        idx_unlabel = idx_not_train[torch.randperm(idx_not_train.size(0))][:batch_size]
        idx_batch = torch.cat((idx_train, idx_unlabel))
        num_train = idx_train.shape[0]

        feat, output = model(x_prop[idx_batch])
        ce_loss = F.nll_loss(output[:num_train], labels[idx_train])
        loss_train = ce_loss

        feat_aug, output_aug = model(x_prop_aug[idx_batch])

        if args.lambda_ce_aug > 0:
            ce_loss_aug = F.nll_loss(output_aug[:num_train], labels[idx_train])
            loss_train += ce_loss_aug * args.lambda_ce_aug

        if args.lambda_pa > 0:
            output_exp = torch.exp(output)
            confidences = output_exp.max(1)[0]
            pseudo_labels = output_exp.max(1)[1].type_as(labels)
            pseudo_labels[:num_train] = labels[idx_train]
            confidences[:num_train] = 1.0

            proto_aug = get_proto_norm_weighted(num_classes, feat_aug, pseudo_labels, confidences)
            proto = get_proto_norm_weighted(num_classes, feat, pseudo_labels, confidences)

            loss_pa = proto_align_loss(proto_aug, proto)
            loss_train += loss_pa * args.lambda_pa

        model.zero_grad()
        loss_train.backward()
        optimizer.step()

    def train_base():
        feat, output = model(x_prop_train)
        ce_loss = F.nll_loss(output, labels_train)
        loss_train = ce_loss

        model.zero_grad()
        loss_train.backward()
        
        optimizer.step()

    t = time.time()
    for epoch in range(args.epochs):
        if args.lambda_ce_aug > 0 or args.lambda_pa > 0:
            if args.batch_size == 0:
                train_full_batch()
            else:
                train_mini_batch(args.batch_size)
        else:
            train_base()
        
        
        model.eval()
        
        _, output = model(x=x,adj=normaliz_dad)

        loss_val = (F.nll_loss(output[idx_val], labels[idx_val])).item()
        acc_val = accuracy(output[idx_val], labels[idx_val]).item()
        if early_stopping.check([acc_val, loss_val], epoch):
            break
        acc_test = accuracy(output[idx_test], labels[idx_test]).item()

        if epoch % 1 == 0:
            current_time = time.time()
            tt = current_time - t
            t = current_time
            print('epoch:{} , acc_val:{:.4f} , acc_test:{:.4f}, time:{:.4f}s'.format(epoch, acc_val, acc_test, tt))

    print('Loading {}th epoch'.format(early_stopping.best_epoch))
    model.load_state_dict(early_stopping.best_state)
    test_acc = test(labels)

    return test_acc

if __name__ == "__main__":

    accs = []
    for trial in range(1, args.num_trials + 1):
        setup_seed(trial)
        test_acc = main()
        print('Trial:{}, Test_acc:{:.4f}'.format(trial, test_acc))
        accs.append(test_acc)

    avg_acc = np.mean(accs) * 100
    std_acc = np.std(accs) * 100
    print('[FINAL RESULT] AVG_ACC:{:.2f}+-{:.2f}'.format(avg_acc, std_acc))