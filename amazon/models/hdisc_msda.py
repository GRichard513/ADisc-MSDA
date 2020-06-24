import numpy as np 
import pandas as pd 
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F

mse = torch.nn.MSELoss()


def weighted_mse(outputs, target, alpha):
    """
    Spectral Norm Loss between one source and one target: ||output^T*output-target^T*target||_2
    Inputs:
        - output: torch.Tensor, source distribution
        - target: torch.Tensor, target distribution
    Output:
        - loss: float, value of the spectral norm of the difference of covariance matrix
        
    """
    loss = torch.sum(torch.stack([alpha[i]*mse(outputs[i], target[i]) for i in range(len(outputs))]))
    return loss

    
class Disc_MSDANet(nn.Module):
    """
    Multi-Source Domain Adaptation with Discrepancy: adapts from multi-source with the hDiscrepancy 
    Learns both a feature representation and weight alpha
    params:
        - 'input_dim': input dimension
        - 'hidden_layers': list of number of neurons in each layer
        - 'output_dim': output dimension (1 in general)
    """
    def __init__(self, params):
        super(Disc_MSDANet, self).__init__()
        self.input_dim = params["input_dim"]
        self.output_dim = params['output_dim']
        self.n_sources = params['n_sources']
        # Parameters of hidden, fully-connected layers, feature learning component.
        self.feature_extractor = params['feature_extractor']
        # Parameter of the final regressor.
        self.h_pred = params['h_pred']
        self.h_disc = params['h_disc']
        self.loss = params['loss']
        self.weighted_loss = params['weighted_loss']
        self.min_pred = params['min_pred']
        self.max_pred = params['max_pred']
        #Parameter
        self.register_parameter(name='alpha', param=torch.nn.Parameter(torch.Tensor(np.ones(self.n_sources)/self.n_sources)) )

    def optimizers(self, opt_feat, opt_pred, opt_disc, opt_alpha):
        """
        Defines optimizers for each parameter
        """
        self.opt_feat = opt_feat
        self.opt_pred = opt_pred
        self.opt_disc = opt_disc
        self.opt_alpha = opt_alpha
        
    def reset_grad(self):
        """
        Set all gradients to zero
        """
        self.opt_feat.zero_grad()
        self.opt_pred.zero_grad()
        self.opt_disc.zero_grad()
        self.opt_alpha.zero_grad()

    def extract_features(self, x):
        z = x.clone()
        for hidden in self.feature_extractor:
            z = hidden(z)
        return z
    def forward(self, X_s, X_t):
        """
        Forward pass
        Inputs:
            - X_s: list of torch.Tensor (m_s, d), source data
            - X_t: torch.Tensor (n, d), target data
        Outputs:
            - y_spred: list of torch.Tensor (m_s), h source prediction
            - y_sdisc: list of torch.Tensor (m_s), h' source prediction
            - y_tpred: list of torch.Tensor (m_s), h target prediction
            - y_tdisc: list of torch.Tensor (m_s), h' target prediction
        """
        # Feature extractor
        sx, tx = X_s.copy(), X_t.clone()
        for i in range(self.n_sources):
            for hidden in self.feature_extractor:
                sx[i] = hidden(sx[i])
        for hidden in self.feature_extractor:
            tx = hidden(tx)
            
        # Predictor h
        y_spred = []
        for i in range(self.n_sources):
            y_sx = sx[i].clone()
            for hidden in self.h_pred:
                y_sx = hidden(y_sx)
            y_spred.append(self.clamp(y_sx))
    
        y_tx = tx.clone()
        for hidden in self.h_pred:
            y_tx = hidden(y_tx)
        y_tpred = self.clamp(y_tx)
            
        # Discrepant h'
        y_sdisc = []
        for i in range(self.n_sources):
            y_tmp = sx[i].clone()
            for hidden in self.h_disc:
                y_tmp = hidden(y_tmp)
            y_sdisc.append(self.clamp(y_tmp))
        y_tmp = tx.clone()
        for hidden in self.h_disc:
            y_tmp = hidden(y_tmp)
        y_tdisc = self.clamp(y_tmp)
        return y_spred, y_sdisc, y_tpred, y_tdisc
    
    def train_prediction(self, X_s, X_t, y_s, clip=1, pred_only=False):
        """
        Train phi and h to minimize the source error
        Inputs:
            - X_s: list of torch.Tensor, source data
            - X_t: torch.Tensor, target data
            - y_s: list of torch.Tensor, source y
            - clip: max values of the gradients
        - Outputs:
        """
        #Training
        self.train()

        #Prediction training
        y_spred, y_sdisc, y_tpred, y_tdisc = self.forward(X_s, X_t)
        loss_pred = self.weighted_loss(y_s, y_spred, self.alpha)
        self.reset_grad()
        loss_pred.backward(retain_graph=True)
        #Clip gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(),clip)
        #Optimization step
        self.opt_pred.step()
        if not pred_only:
            self.opt_feat.step()
        self.reset_grad()
        return loss_pred
    
    def train_h_disc_pred(self, X_s, X_t, y_s, clip=1):
        self.train()
        #Discrepancy training
        y_spred, y_sdisc, y_tpred, y_tdisc = self.forward(X_s, X_t)
        disc = self.weighted_loss(y_s, y_sdisc, self.alpha)
        loss_disc = disc
        self.reset_grad()
        loss_disc.backward(retain_graph=True)
        #Clip gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(),clip)
        #Optimization step
        self.opt_disc.step()
        self.reset_grad()

        
    def train_h_discrepancy(self, X_s, X_t, y_s, clip=1):
        """
        Train h' to maximize the discrepancy
        Inputs:
            - X_s: list of torch.Tensor, source data
            - X_t: torch.Tensor, target data
            - y_s: list of torch.Tensor, source y
            - clip: max values of the gradients
        """
        #Training
        self.train()

        #Discrepancy training
        y_spred, y_sdisc, y_tpred, y_tdisc = self.forward(X_s, X_t)
        disc = self.weighted_loss(y_s, y_sdisc, self.alpha)-self.loss(y_tpred, y_tdisc)
        #source_loss = self.weighted_loss(y_s, y_sdisc, self.alpha)
        loss_disc = disc
        self.reset_grad()
        loss_disc.backward(retain_graph=True)
        #Clip gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(),clip)
        #Optimization step
        self.opt_disc.step()
        self.reset_grad()

    def train_feat_discrepancy(self, X_s, X_t, y_s, clip=1, mu = 1):
        """
        Train phi to minimize the discrepancy
        Inputs:
            - X_s: list of torch.Tensor, source data
            - X_t: torch.Tensor, target data
            - y_s: list of torch.Tensor, source y
            - clip: max values of the gradients
        """
        #Training
        self.train()

        #Feature training
        y_spred, y_sdisc, y_tpred, y_tdisc = self.forward(X_s, X_t)            
        disc = torch.abs(self.weighted_loss(y_spred, y_sdisc, self.alpha)-self.loss(y_tpred, y_tdisc))
        source_loss = self.weighted_loss(y_s, y_spred, self.alpha)
        loss = disc + source_loss*mu
        self.reset_grad()
        loss.backward(retain_graph=True)
        #Clip gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(),clip)
        #Optimization step
        self.opt_feat.step()
        self.reset_grad()

    def train_alpha_discrepancy(self, X_s, X_t, y_s, clip=1, lam_alpha=0.01):
        """
        Train alpha to minimize the discrepancy
        Inputs:
            - X_s: list of torch.Tensor, source data
            - X_t: torch.Tensor, target data
            - y_s: list of torch.Tensor, source y
            - clip: max values of the gradients
        """
        #Training
        self.train()

        #Feature training
        y_spred, y_sdisc, y_tpred, y_tdisc = self.forward(X_s, X_t)
        disc = torch.abs(self.loss(y_tpred, y_tdisc)-self.weighted_loss(y_spred, y_sdisc, self.alpha))
        loss_disc = disc + lam_alpha*torch.norm(self.alpha, p=2) 
        self.reset_grad()
        loss_disc.backward(retain_graph=True)
        #Clip gradients
        torch.nn.utils.clip_grad_norm_(self.alpha,clip)
        #Optimization step
        self.opt_alpha.step()
        self.reset_grad()
        #Normalization (||alpha||_1=1)
        with torch.no_grad():
            self.alpha.clamp_(1/(self.n_sources*10),1-1/(self.n_sources*10))
            self.alpha.div_(torch.norm(F.relu(self.alpha), p=1))
    
    def predict(self, X):
        z = X.clone()
        for hidden in self.feature_extractor:
            z = hidden(z)
        for hidden in self.h_pred:
            z = hidden(z)
        return self.clamp(z)
    
    def clamp(self, x):
        return torch.clamp_(x, self.min_pred, self.max_pred)
    
    def compute_loss(self, X_s, X_t, y_s):
        """
        Compute the losses
        """
        y_spred, y_sdisc, y_tpred, y_tdisc = self.forward(X_s, X_t)            
        source_loss = self.weighted_loss(y_s, y_spred, self.alpha)
        disc = torch.abs(self.loss(y_tpred, y_tdisc)-self.weighted_loss(y_spred, y_sdisc, self.alpha))
        return source_loss, disc
        
