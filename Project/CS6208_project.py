import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch_geometric.nn import GCNConv, TopKPooling, global_mean_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import argparse

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from torch_geometric.data import DataLoader
from celeb_a_graph_dataloader import CS6208_dataset
from model import GCN
from sklearn.metrics import average_precision_score
from distutils.util import strtobool

from ECE import *

import pdb
import warnings
warnings.filterwarnings('ignore')
import pickle


attributes = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick','Wearing_Necklace', 'Wearing_Necktie' ,'Young']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expt_name', type=str, default="GCNN Faces", help='Experiment name')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer, adam or sgd.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate for sgd.')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for sgd')
    parser.add_argument('--workers', default=2, type=int, help='Number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=100, help='Total training epochs.')
    parser.add_argument('--seed', type=int, default=0, help='Random Seed')

    parser.add_argument('--dataset', type=str, default="x5_x5", help='Dataset modes')
    parser.add_argument('--alpha', type=float, default=0.0, help='Label smoothing')
    parser.add_argument('--gamma', type=float, default=2.0, help='Focal factor')
    parser.add_argument('--constraints', type=int, default=0, help='Max Ent mode constraints 1-Mu 2-Variance 3-Poly')

    return parser.parse_args()

def save_checkpoint(state, epoch, expt_name):
    print("==> Checkpoint saved")
    
    if not os.path.exists('./models/' + expt_name):
        os.makedirs('./models/' + expt_name)
        
    outfile = './models/' + expt_name + '/' + str(epoch) + '_' + expt_name + '.pth.tar'
    torch.save(state, outfile)
    
def load_checkpoint(model, optimizer, weight_file):
    print("==> Loading Checkpoint: " + weight_file)
    
    #weight_file = r'checkpoint.pth.tar'
    if torch.cuda.is_available() == False:
        checkpoint = torch.load(weight_file, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(weight_file)
        
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

def check_MAP_scores(cat_labels, cat_preds, categories=None):
    
    # Sklearn usage: average_precision_score(y_true, y_scores, average=None) # returns ap for each attribute
    ap = average_precision_score(cat_labels, cat_preds, average=None)
    #print("Mean AP:", ap.mean(), mean_ap)
    
    #print(len(cat_labels))
    cat_ap = {}
    for i in range(len(categories)):
        cat_ap[categories[i]] = ap[i]

    #    print('Category: %16s %.5f' % (categories[i], ap[i]))
    #print('====')
    
    return cat_ap, ap.mean()

def solve_mean_lagrange(x, mu, lam=0, max_iter = 20, tol = 1e-15):
    #Implements the Newton Raphson method:
    i = 0
    old_lam = lam

    def fx_mean(lam1, x, mu):
        return mu/np.exp(-1) - np.dot( x , np.exp(-lam1 * x) )

    def dxfx_mean(lam1, x):
        return np.dot( x**2 , np.exp(-lam1 * x)) + np.exp(-lam1 * x).sum()

    while abs( fx_mean(lam, x, mu) ) > tol: #run the helper function and check

        lam = old_lam - fx_mean(lam, x, mu)/dxfx_mean(lam,x)  # Newton-Raphson equation
        #print("Iteration" + str(i) + ": x = " + str(lam) + ", f(x) = " +  str( fx(lam, x, mu) ) )  
          
        old_lam = lam
        i += 1
        
        if i > max_iter:
          break

    return  torch.tensor(lam)

class MaxEntLoss(nn.Module):
    def __init__(self, ratio, constraints, gamma=2, num_classes=10, eps=1e-7):
        super(MaxEntLoss, self).__init__()

        self.gamma = gamma
        self.eps = eps

        self.constraints = constraints
        x = torch.tensor(range(num_classes), dtype=float)
        
        self.target_mu = (1 - ratio) #expectation for each attribute
        #target_var = torch.sum(ratio * x.pow(2)) - target_mu.pow(2)

        self.lam_1 = torch.zeros(len(ratio))

        for i, mu in enumerate(self.target_mu):
            self.lam_1[i] = solve_mean_lagrange(x, mu)

        self.pos_ratio = ratio
        self.neg_ratio = 1 - ratio
        self.target_mu = self.target_mu.to("cuda")
        self.lam_1 = self.lam_1.to("cuda")
    def forward(self, p, y, weights):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-class binarized vector)
        """

        batch_sz = y.shape[0]
        
        # Basic BCE computation
        los_pos = y * torch.log(p.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log( (1 - p).clamp(min=self.eps))

        mu = (1 - p)
        #var = torch.sum(p * self.x.pow(2), dim=1) - mu.pow(2)

        # Focal loss
        if self.gamma > 0:
            focal_pos = (1 - p).pow(self.gamma)
            los_pos *= focal_pos

            focal_neg = p.pow(self.gamma)
            los_neg *= focal_neg

        if weights != None:
            loss = weights * (los_pos + los_neg)
        else:
            loss = los_pos + los_neg

        if self.constraints == 1: #Exponential Distribution (Mean constraint)
        
            mu_loss = torch.abs(mu - self.target_mu)
            loss = -torch.sum(loss, dim=1) + (self.lam_1 * mu_loss).sum(1).mean()
        else:

            loss = -torch.sum(loss, dim=1)

        return loss.mean()

def test_data(model, data_loader):
    
    ce_losses = []
    categorical_predictions = []
    categorical_labels = []

    with torch.no_grad():
        model.eval()

        for batch in data_loader:
            batch.to("cuda")

            probs, _ = model(batch.x.float(), batch.edge_index, batch.batch)
            targets = batch.y.reshape(probs.shape).float()

            categorical_predictions.append(probs.detach().cpu() )
            categorical_labels.append(targets.detach().cpu() )

    
        cat_predictions = np.concatenate(categorical_predictions, axis=0) #N x 40
        cat_labels = np.concatenate(categorical_labels, axis=0) #N x 40

        #MAP = average_precision_score(cat_labels, cat_predictions, average=None).mean()
        AP, MAP = check_MAP_scores(cat_labels, cat_predictions, categories=attributes)

        mean_ECE = np.zeros(len(attributes))
        for i in range(len(attributes)):

            confidence_vals_list = cat_predictions[:,i]
            predictions_list = np.round(cat_predictions[:,i])
            labels_list = cat_labels[:,i]

            ECE = expected_calibration_error(confidence_vals_list, predictions_list, labels_list, num_bins=15)
            mean_ECE[i] = ECE


    return AP, MAP, mean_ECE.mean()

def run_training():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cpu = torch.device('cpu')

    args = parse_args()
    torch.manual_seed(args.seed)
    expt_name = args.expt_name

    if args.dataset == "x5_x5":
        ID = "x5"
        OOD = "x5"

    elif args.dataset == "x68_x68":
        ID = "x68"
        OOD = "x68"

    elif args.dataset == "x5_x68":
        ID = "x5"
        OOD = "x68"

    elif args.dataset == "x68_x5":
        ID = "x68"
        OOD = "x5"

    train_df = pd.read_pickle("./datasets/" + ID + '_train_landmarks.pkl')
    valid_df = pd.read_pickle("./datasets/" + OOD + '_valid_landmarks.pkl')
    test_df = pd.read_pickle("./datasets/" + OOD + '_test_landmarks.pkl')
    
    train_dataset = CS6208_dataset(root='.', dataframe=train_df, DLIB=ID, data_partition='train')
    valid_dataset = CS6208_dataset(root='.', dataframe=valid_df, DLIB=OOD, data_partition='valid')
    test_dataset = CS6208_dataset(root='.', dataframe=test_df, DLIB=OOD, data_partition='test')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    model = GCN(num_attributes=40)
    model.to(device)
    print(model)
    print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    prior = torch.tensor([0.1117, 0.2659, 0.5136, 0.2045, 0.0228, 0.1517, 0.2409, 0.2356, 0.2390,
                    0.1491, 0.0514, 0.2039, 0.1437, 0.0577, 0.0465, 0.0646, 0.0635, 0.0424,
                    0.3843, 0.4524, 0.4194, 0.4822, 0.0408, 0.1159, 0.8342, 0.2832, 0.0430,
                    0.2755, 0.0801, 0.0647, 0.0563, 0.4797, 0.2086, 0.3194, 0.1865, 0.0494,
                    0.4696, 0.1214, 0.0730, 0.7789])
    
    criterion = MaxEntLoss(ratio=prior, constraints=args.constraints, num_classes=2, gamma=args.gamma)

    best_valid_MAP = 0
    for epoch in tqdm(range(args.epochs), desc="Training:"):
        losses = []

        categorical_predictions = []
        categorical_labels = []

        for batch in train_loader:
            batch.to(device)

            probs, _ = model(batch.x.float(), batch.edge_index, batch.batch) # Passing the node features and the connection info
            targets = batch.y.reshape(probs.shape).float()

            count = targets.sum(0).unsqueeze(0).cpu() # number of positives in each batch
            f_count = args.batch_size - count
            max_count = torch.cat([count, f_count]).max(0)[0]

            weights = torch.ones((1,40))
            weights[count != 0] = (max_count/count)[count!=0] 
            weights[count == 0] = (max_count/f_count)[count==0]  # maximum penalty since it did not appear

            #loss = F.binary_cross_entropy(probs, targets, weights.to(device)) #Bx40
            loss = criterion(probs, targets, weights.to(device))
            #pdb.set_trace()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss)
        
            categorical_predictions.append(probs.detach().cpu() )
            categorical_labels.append(targets.detach().cpu() )

    
        cat_predictions = np.concatenate(categorical_predictions, axis=0) #N x 40
        cat_labels = np.concatenate(categorical_labels, axis=0) #N x 40

        training_MAP = average_precision_score(cat_labels, cat_predictions, average=None).mean()
        _, valid_MAP, valid_ECE = test_data(model, valid_loader)
        print("Epoch", epoch, 
            "Loss", sum(losses)/len(losses), 
            "Train MAP", training_MAP, 
            "Valid MAP", valid_MAP,
            "Valid ECE", valid_ECE
            )

        if not os.path.exists('./logs/' + expt_name):
            os.makedirs('./logs/' + expt_name)

        with open('./logs/' + expt_name + '/' + expt_name + '_scores.txt', "a") as f:
            print(training_MAP, valid_MAP, valid_ECE, file=f)

        if valid_MAP > best_valid_MAP and epoch != 0:
            best_valid_MAP = valid_MAP
            checkpoint = {'state_dict': model.state_dict(),'optimizer': optimizer.state_dict()}
            save_checkpoint(checkpoint, 'best', expt_name)

    #Evaluate the best performing model
    best_model_path = './models/' + expt_name + '/' + 'best_' + expt_name + '.pth.tar'
    best_model = load_checkpoint(model, optimizer, best_model_path)
    test_AP, test_MAP, test_ECE = test_data(model, test_loader)
    
    with open(expt_name + '_test_scores.txt', "a") as f:
        print(test_AP, test_MAP, test_ECE, file=f)

    print("Test MAP", test_AP, test_MAP, test_ECE)

    with open(expt_name + '_test_AP.pkl', 'wb') as handle:
        pickle.dump(test_AP, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    run_training()
    


