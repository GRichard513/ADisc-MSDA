#### Script versions of the notebook ####


import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import importlib
from IPython.display import display
import torch
from models.hdisc_msda import Disc_MSDANet, weighted_mse
from models.MDAN import MDANet_general
from utils.load_amazon import load_amazon
from utils.utils_hdisc import batch_loader, split_source_target, val_split, save_result
import h5py


##Data Loading##

np.random.seed(0)
torch.manual_seed(0)
#Loading
X_amazon, y_amazon, domain_list = load_amazon(domains=None)
#Standardize labels
mu_y = np.mean(np.concatenate(y_amazon))
std_y = np.std(np.concatenate(y_amazon))
y_amazon = [(y-mu_y)/std_y for y in y_amazon]
#y_amazon = [y-3 for y in y_amazon]
#Number of domains
n_domains = len(X_amazon)
print(n_domains)


##Base model##
def get_feature_extractor():
    return nn.ModuleList([
            nn.Linear(X_amazon[0].shape[1], 500, bias=False), nn.LeakyReLU(), nn.Dropout(p=0.1),
            nn.Linear(500, 20, bias=False), nn.LeakyReLU(), nn.Dropout(p=0.1)])

def get_predictor(output_dim=1):
    return  nn.ModuleList([
            #nn.Linear(500,100, bias=False), nn.ELU(), nn.Dropout(p=0.1),
            nn.Linear(20, output_dim, bias=False)])

def get_discriminator(output_dim=1):
    return nn.ModuleList([
            #nn.Linear(500, 100, bias=False), nn.ELU(), nn.Dropout(p=0.1),
            nn.Linear(20, output_dim, bias=False)])
    
##1. MLP##

#Number of experiments to launch
nb_experiments = 5
results_mse, results_mae = [], []

params= {'input_dim': X_amazon[0].shape[1], 'output_dim': 1, 'n_sources': n_domains-1, 'loss': torch.nn.MSELoss(),
         'weighted_loss': weighted_mse, 'min_pred': -np.inf, 'max_pred': np.inf}
#Hyperparameters
epochs = 100
device = torch.device('cuda:0')
lr = 0.001
batch_size = 128
keep_best = True
current_loss = 10e9
for exp in range(nb_experiments):
    print('\n ----------------------------- %i / %i -----------------------------'%(exp+1, nb_experiments))
    mse_list, mae_list =  {}, {}
    for i in range(len(domain_list)):
        domain = domain_list[i]
        #Split source and target
        torch.cuda.empty_cache() 
        X_s, X_t, y_s, y_t = split_source_target(X_amazon, y_amazon, i, device, merge=False)
        #Validation split
        X_train, X_val, y_train, y_val = val_split(X_s, y_s)
        #Initialize model
        params['feature_extractor'] = get_feature_extractor()
        params['h_pred'] = get_predictor(output_dim=1)
        params['h_disc'] = get_discriminator(output_dim=1)
        model = Disc_MSDANet(params).to(device)
        opt_feat = torch.optim.Adam([{'params': model.feature_extractor.parameters()}],lr=lr)
        opt_pred = torch.optim.Adam([{'params': model.h_pred.parameters()}],lr=lr)
        opt_disc =torch.optim.Adam([{'params': model.h_disc.parameters()}],lr=lr)
        opt_alpha =torch.optim.Adam([{'params': model.alpha}],lr=lr)
        model.optimizers(opt_feat, opt_pred, opt_disc, opt_alpha)
        print('----', domain, '----')
        for epoch in range(epochs):
            model.train()
            loader = batch_loader(X_train, y_train ,batch_size = batch_size)
            for x_bs, y_bs in loader:
                loss_pred = model.train_prediction(x_bs, X_t, y_bs, clip=1, pred_only=False)
            #Validation
            model.eval()
            val_loss, _ = model.compute_loss(X_val, X_t, y_val)
            if (val_loss.item()<current_loss)*(keep_best):
                current_loss = val_loss.item()
            if (epoch+1)%100==0:
                model.eval()
                source_loss, disc = model.compute_loss(X_s, X_t, y_s)
                reg_loss = model.loss(y_t, model.predict(X_t))
                print('Epoch: %i/%i ; Train loss: %.3f ; Validation loss: %.3f Test loss: %.3f'%(epoch+1, epochs, source_loss.item(), val_loss.item(), reg_loss.item()))
        mse_list[domain] = model.loss(y_t, model.predict(X_t)).item()
        mae_list[domain] = torch.sum(torch.abs(y_t.squeeze_()- model.predict(X_t).squeeze_()))/y_t.shape[0]
    results_mse.append(mse_list)
    results_mae.append(mae_list)
save_result(results_mse, './results/1SRC_noadapt_mse.csv', domain_list)
save_result(results_mae, './results/1SRC_noadapt_mae.csv', domain_list)


##2. DANN##

#Number of experiments to launch
nb_experiments = 5
results_mse, results_mae = [], []

configs = {"num_epochs": 100, "num_domains": n_domains-1, 
           "mode": 'DANN', "verbose": 2}


#Hyperparameters
epochs = 100
device = torch.device('cuda:0')
lr = 0.001
batch_size = 128
mu = 0.01
gamma = 10
mode = configs['mode']

for exp in range(nb_experiments):
    print('\n ----------------------------- %i / %i -----------------------------'%(exp+1, nb_experiments))
    mse_list, mae_list =  {}, {}
    for i in range(len(domain_list)):
        domain = domain_list[i]
        #Split source and target
        torch.cuda.empty_cache() 
        X_s, X_t, y_s, y_t = split_source_target(X_amazon, y_amazon, i, device, merge=False)
        #Initialize model
        configs['hiddens'] = get_feature_extractor()
        configs['predictor'] = get_predictor(output_dim=1)
        configs['discriminator'] = [get_discriminator(output_dim=2) for _ in range(configs['num_domains'])]
        mdan = MDANet_general(configs).to(device)
        for i in range(configs['num_domains']):
            mdan.discriminator[i].to(device)
        optimizer = torch.optim.Adam(mdan.parameters(), lr=lr)
        print('----', domain, '----')
        for epoch in range(epochs):
            mdan.train()
            loader = batch_loader(X_s, y_s ,batch_size = batch_size)
            for x_bs, y_bs in loader:
                slabels = torch.ones(batch_size, requires_grad=False).type(torch.LongTensor).to(device)
                tlabels = torch.zeros(batch_size, requires_grad=False).type(torch.LongTensor).to(device)
                ridx = np.random.choice(X_t.shape[0], batch_size)
                #Batch selection of X_t
                tinputs = X_t[ridx, :]
                optimizer.zero_grad()
                vals, sdomains, tdomains = mdan(x_bs, tinputs)
                # Compute prediction accuracy on multiple training sources.
                losses = torch.stack([F.mse_loss(vals[j], y_bs[j]) for j in range(configs['num_domains'])])
                domain_losses = torch.stack([F.nll_loss(sdomains[j], slabels) +
                                           F.nll_loss(tdomains[j], tlabels) for j in range(configs['num_domains'])])
                # Different final loss function depending on different training modes.
                if mode == "maxmin":
                    loss = torch.max(losses) + mu * torch.min(domain_losses)
                elif mode == "dynamic":
                    loss = torch.log(torch.sum(torch.exp(gamma * (losses + mu * domain_losses)))) / gamma
                elif mode=='no-weight':
                    loss = torch.mean(losses)
                elif mode=='DANN':
                    loss = torch.mean(losses) + mu * torch.mean(domain_losses)
                loss.backward(retain_graph=True )
                optimizer.step()
            if (epoch+1)%100==0:
                mdan.eval()
                preds_labels = mdan.inference(X_t)
                source_loss = torch.mean(torch.stack([torch.nn.MSELoss()(y_s[i], mdan.inference(X_s[i])) for i in range(len(y_s))]))
                print('Epoch: %i ; Train loss: %.3f ;  Loss: %.3f'%(epoch+1, source_loss, torch.nn.MSELoss()(y_t, preds_labels)))
        mse_list[domain] = torch.nn.MSELoss()(y_t, mdan.inference(X_t)).item()
        mae_list[domain] = torch.sum(torch.abs(y_t.squeeze_()- mdan.inference(X_t).squeeze_()))/y_t.shape[0]
    results_mse.append(mse_list)
    results_mae.append(mae_list)
    
save_result(results_mse, './results/1SRC_DANN_mse.csv', domain_list)
save_result(results_mae, './results/1SRC_DANN_mae.csv', domain_list)


##3. AHDA##

#Number of experiments to launch
nb_experiments = 5
results_mse, results_mae = [], []

params= {'input_dim': X_amazon[0].shape[1], 'output_dim': 1, 'n_sources': n_domains-1, 'loss': torch.nn.MSELoss(),
         'weighted_loss': weighted_mse, 'min_pred': -np.inf, 'max_pred': np.inf}
#Number of epochs
epochs_pretrain = 0
epochs_adapt = 100
epochs_h_disc, epochs_feat, epochs_alpha, epochs_pred = 1, 1, 1, 1
device = torch.device('cuda:1')
lr = 0.001
batch_size = 128
for exp in range(nb_experiments):
    print('\n ----------------------------- %i / %i -----------------------------'%(exp+1, nb_experiments))
    mse_list, mae_list =  {}, {}
    for i in range(len(domain_list)):
        domain = domain_list[i]
        #Split source and target
        torch.cuda.empty_cache() 
        X_s, X_t, y_s, y_t = split_source_target(X_amazon, y_amazon, i, device, merge=False)
        #Merge all sources
        #Initialize model
        params['feature_extractor'] = get_feature_extractor()
        params['h_pred'] = get_predictor(output_dim=1)
        params['h_disc'] = get_discriminator(output_dim=1)
        model = Disc_MSDANet(params).to(device)
        opt_feat = torch.optim.Adam([{'params': model.feature_extractor.parameters()}],lr=lr)
        opt_pred = torch.optim.Adam([{'params': model.h_pred.parameters()}],lr=lr)
        opt_disc =torch.optim.Adam([{'params': model.h_disc.parameters()}],lr=lr)
        opt_alpha =torch.optim.Adam([{'params': model.alpha}],lr=lr)
        model.optimizers(opt_feat, opt_pred, opt_disc, opt_alpha)
        print('----', domain, '----')
        #Pre-training
        print('------------Pre-training------------')
        for epoch in range(epochs_pretrain):
            loader = batch_loader(X_s, y_s ,batch_size = batch_size)
            for x_bs, y_bs in loader:
                loss_pred = model.train_prediction(x_bs, X_t, y_bs, clip=1, pred_only=False)
            if (epoch+1)%10==0:
                source_loss, disc = model.compute_loss(X_s, X_t, y_s)
                reg_loss = model.loss(y_t, model.predict(X_t))
                print('Epoch: %i/%i ; Train loss: %.3f ; Disc: %.3f ; Test loss: %.3f'%(epoch+1, epochs_pretrain, source_loss.item(), disc.item(), reg_loss.item()))

        #Alternated training
        print('------------Alternated training------------')
        for epoch in range(epochs_adapt):
            loader = batch_loader(X_s, y_s ,batch_size = batch_size)
            for x_bs, y_bs in loader:
                ridx = np.random.choice(X_t.shape[0], batch_size)
                x_bt = X_t[ridx,:]
                #Train h to minimize source loss
                for e in range(epochs_pred):
                    model.train_prediction(x_bs, x_bt, y_bs, pred_only=False)
                
                #Train h' to maximize discrepancy
                for e in range(epochs_h_disc):
                    model.train_h_discrepancy(x_bs, x_bt, y_bs)

                #Train phi to minimize discrepancy
                for e in range(epochs_feat):
                    model.train_feat_discrepancy(x_bs, x_bt, y_bs, mu=0.001)
                
            #Logs
            if (epoch+1)%100==0:
                source_loss, disc = model.compute_loss(X_s, X_t, y_s)
                reg_loss = model.loss(y_t, model.predict(X_t))
                print('Epoch: %i/%i (h_pred); Train loss: %.3f ; Disc: %.3f ; Test loss: %.3f'%(epoch+1, epochs_adapt, source_loss.item(), disc.item(), reg_loss.item()))
        mse_list[domain] = model.loss(y_t, model.predict(X_t)).item()
        mae_list[domain] = torch.sum(torch.abs(y_t.squeeze_()- model.predict(X_t).squeeze_()))/y_t.shape[0]
    results_mse.append(mse_list)
    results_mae.append(mae_list)
    
save_result(results_mse, './results/1SRC_Adisc_mse.csv', domain_list)
save_result(results_mae, './results/1SRC_Adisc_mae.csv', domain_list)


##4. MDAN##
#Number of experiments to launch
nb_experiments = 5
results_mse, results_mae = [], []

configs = {"num_domains": len(domain_list)-1, 
           "mode": 'dynamic', "verbose": 2}

#Hyperparameters
epochs = 100
device = torch.device('cuda:1')
lr = 0.001
batch_size = 128
mu = 0.1
gamma = 10
mode = 'dynamic'

for exp in range(nb_experiments):
    print('\n ----------------------------- %i / %i -----------------------------'%(exp+1, nb_experiments))
    mse_list, mae_list =  {}, {}
    alphas = {}
    for i in range(len(domain_list)):
        domain = domain_list[i]
        #Split source and target
        torch.cuda.empty_cache() 
        X_s, X_t, y_s, y_t = split_source_target(X_amazon, y_amazon, i, device, merge=False)
        #Initialize model
        configs['hiddens'] = get_feature_extractor()
        configs['predictor'] = get_predictor(output_dim=1)
        configs['discriminator'] = [get_discriminator(output_dim=2) for _ in range(len(domain_list)-1)]
        mdan = MDANet_general(configs).to(device)
        for i in range(configs['num_domains']):
            mdan.discriminator[i].to(device)
        optimizer = torch.optim.Adam(mdan.parameters(), lr=lr)
        print('----', domain, '----')
        for epoch in range(epochs):
            mdan.train()
            loader = batch_loader(X_s, y_s ,batch_size = batch_size)
            for x_bs, y_bs in loader:
                slabels = torch.ones(batch_size, requires_grad=False).type(torch.LongTensor).to(device)
                tlabels = torch.zeros(batch_size, requires_grad=False).type(torch.LongTensor).to(device)
                ridx = np.random.choice(X_t.shape[0], batch_size)
                #Batch selection of X_t
                tinputs = X_t[ridx, :]
                optimizer.zero_grad()
                vals, sdomains, tdomains = mdan(x_bs, tinputs)
                # Compute prediction accuracy on multiple training sources.
                losses = torch.stack([F.mse_loss(vals[j], y_bs[j]) for j in range(configs['num_domains'])])
                domain_losses = torch.stack([F.nll_loss(sdomains[j], slabels) +
                                           F.nll_loss(tdomains[j], tlabels) for j in range(configs['num_domains'])])
                # Different final loss function depending on different training modes.
                if mode == "maxmin":
                    loss = torch.max(losses) + mu * torch.min(domain_losses)
                elif mode == "dynamic":
                    loss = torch.log(torch.sum(torch.exp(gamma * (losses + mu * domain_losses)))) / gamma
                elif mode=='no-weight':
                    loss = torch.mean(losses)
                elif mode=='DANN':
                    loss = torch.mean(losses) + mu * torch.mean(domain_losses)
                loss.backward(retain_graph=True )
                optimizer.step()
            if (epoch+1)%100==0:
                mdan.eval()
                preds_labels = mdan.inference(X_t)
                source_loss = torch.mean(torch.stack([torch.nn.MSELoss()(y_s[i], mdan.inference(X_s[i])) for i in range(len(y_s))]))
                print('Epoch: %i ; Train loss: %.3f ;  Loss: %.3f'%(epoch+1, source_loss, torch.nn.MSELoss()(y_t, preds_labels)))
        mse_list[domain] = torch.nn.MSELoss()(y_t, mdan.inference(X_t)).item()
        mae_list[domain] = torch.sum(torch.abs(y_t.squeeze_()- mdan.inference(X_t).squeeze_()))/y_t.shape[0]
        alphas[domain] = (torch.exp(gamma * (losses + mu * domain_losses))/torch.sum(torch.exp(gamma * (losses + mu * domain_losses)))).cpu().detach().numpy()
    results_mse.append(mse_list)
    results_mae.append(mae_list)
    
save_result(results_mse, './results/MDAN_mse.csv', domain_list)
save_result(results_mae, './results/MDAN_mae.csv', domain_list)
w = h5py.File('./results/MDAN_alpha.h5', 'w')
for key, val in alphas.items():
    w.create_dataset(name=key, data=val)
w.close()

## 5. AHD-MSDA##

import importlib
import models.hdisc_msda 
importlib.reload(models.hdisc_msda)
from models.hdisc_msda import Disc_MSDANet

#Number of experiments to launch
nb_experiments = 5
results_mse, results_mae = [], []

params= {'input_dim': X_amazon[0].shape[1], 'output_dim': 1, 'n_sources': n_domains-1, 'loss': torch.nn.MSELoss(),
         'weighted_loss': weighted_mse, 'min_pred': -np.inf, 'max_pred': np.inf}

#Number of epochs
epochs_pretrain, epochs_adapt = 0, 100
epochs_h_disc, epochs_feat, epochs_alpha, epochs_pred = 2, 1, 1, 1

device = torch.device('cuda:1')
lr = 0.001
batch_size = 128
alphas = {}
for exp in range(nb_experiments):
    print('\n ----------------------------- %i / %i -----------------------------'%(exp+1, nb_experiments))
    mse_list, mae_list =  {}, {}
    alphas = {}
    for i in range(len(domain_list)):
        domain = domain_list[i]
        #Split source and target
        X_s, X_t, y_s, y_t = split_source_target(X_amazon, y_amazon, i, device, merge=False)
        #Merge all sources
        #Initialize model
        params['feature_extractor'] = get_feature_extractor()
        params['h_pred'] = get_predictor(output_dim=1)
        params['h_disc'] = get_discriminator(output_dim=1)
        model = Disc_MSDANet(params).to(device)
        opt_feat = torch.optim.Adam([{'params': model.feature_extractor.parameters()}],lr=lr)
        opt_pred = torch.optim.Adam([{'params': model.h_pred.parameters()}],lr=lr)
        opt_disc = torch.optim.Adam([{'params': model.h_disc.parameters()}],lr=lr)
        opt_alpha = torch.optim.Adam([{'params': model.alpha}],lr=lr)
        model.optimizers(opt_feat, opt_pred, opt_disc, opt_alpha)
        print('----', domain, '----')
        #Pre-training
        print('------------Pre-training------------')
        for epoch in range(epochs_pretrain):
            loader = batch_loader(X_s, y_s ,batch_size = batch_size)
            for x_bs, y_bs in loader:
                loss_pred = model.train_prediction(x_bs, X_t, y_bs, clip=1, pred_only=False)
            if (epoch+1)%1==0:
                source_loss, disc = model.compute_loss(X_s, X_t, y_s)
                reg_loss = model.loss(y_t, model.predict(X_t))
                print('Epoch: %i/%i ; Train loss: %.3f ; Disc: %.3f ; Test loss: %.3f'%(epoch+1, epochs_pretrain, source_loss.item(), disc.item(), reg_loss.item()))

        #Alternated training
        print('------------Alternated training------------')
        for epoch in range(epochs_adapt):
            model.train()
            loader = batch_loader(X_s, y_s ,batch_size = batch_size)
            for x_bs, y_bs in loader:
                ridx = np.random.choice(X_t.shape[0], batch_size)
                x_bt = X_t[ridx,:]
                #Train h to minimize source loss
                for e in range(epochs_pred):
                    model.train_prediction(x_bs, x_bt, y_bs, pred_only=False)

                #Train h' to maximize discrepancy
                for e in range(epochs_h_disc):
                    model.train_h_discrepancy(x_bs, x_bt, y_bs)
                
                #Train phi to minimize discrepancy
                for e in range(epochs_feat):
                    model.train_feat_discrepancy(x_bs, x_bt, y_bs, mu=0)
                    
                #Train alpha to minimize discrepancy
                for e in range(epochs_alpha):
                    model.train_alpha_discrepancy(x_bs, x_bt, y_bs, clip=1, lam_alpha=0.01)

            #Logs
            if (epoch+1)%100==0:
                model.eval()
                source_loss, disc = model.compute_loss(X_s, X_t, y_s)
                reg_loss = model.loss(y_t, model.predict(X_t))
                print('Epoch: %i/%i (h_pred); Train loss: %.3f ; Disc: %.3f ; Test loss: %.3f'%(epoch+1, epochs_adapt, source_loss.item(), disc.item(), reg_loss.item()))
        mse_list[domain] = model.loss(y_t, model.predict(X_t)).item()
        mae_list[domain] = torch.sum(torch.abs(y_t.squeeze_()- model.predict(X_t).squeeze_())).item()/y_t.shape[0]
        alphas[domain] = model.alpha.cpu().detach().numpy()
        print(mae_list[domain], alphas[domain])
    results_mse.append(mse_list)
    results_mae.append(mae_list)
    save_result(results_mse, './results/ADisc_MSDA_mse.csv', domain_list)
    save_result(results_mae, './results/ADisc_MSDA_mae.csv', domain_list)
save_result(results_mse, './results/ADisc_MSDA_mse.csv', domain_list)
save_result(results_mae, './results/ADisc_MSDA_mae.csv', domain_list)
w = h5py.File('./results/ADisc_MSDA_alpha.h5', 'r')
for key, val in alphas.items():
    w.create_dataset(name=k, data=val)
w.close()