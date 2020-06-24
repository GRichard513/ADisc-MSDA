#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


class GradientReversalLayer(torch.autograd.Function):
    """
    Implement the gradient reversal layer for the convenience of domain adaptation neural network.
    The forward part is the identity function while the backward part is the negative function.
    """
    def forward(self, inputs):
        return inputs

    def backward(self, grad_output):
        grad_input = grad_output.clone()
        grad_input = -grad_input
        return grad_input


class MDANet(nn.Module):
    """
    Multi-layer perceptron with adversarial regularizer by domain classification.
    """
    def __init__(self, configs):
        super(MDANet, self).__init__()
        self.input_dim = configs["input_dim"]
        self.num_hidden_layers = len(configs["hidden_layers"])
        self.num_neurons = [self.input_dim] + configs["hidden_layers"]
        self.num_domains = configs["num_domains"]
        # Parameters of hidden, fully-connected layers, feature learning component.
        self.hiddens = nn.ModuleList([nn.Linear(self.num_neurons[i], self.num_neurons[i+1])
                                      for i in range(self.num_hidden_layers)])
        # Parameter of the final softmax classification layer.
        self.softmax = nn.Linear(self.num_neurons[-1], configs["num_classes"])
        # Parameter of the domain classification layer, multiple sources single target domain adaptation.
        self.domains = nn.ModuleList([nn.Linear(self.num_neurons[-1], 2) for _ in range(self.num_domains)])
        # Gradient reversal layer.
        self.grls = [GradientReversalLayer() for _ in range(self.num_domains)]
        
    def forward(self, sinputs, tinputs):
        """
        :param sinputs:     A list of k inputs from k source domains.
        :param tinputs:     Input from the target domain.
        :return:
        """
        sh_relu, th_relu = sinputs, tinputs
        for i in range(self.num_domains):
            for hidden in self.hiddens:
                sh_relu[i] = F.relu(hidden(sh_relu[i]))
        for hidden in self.hiddens:
            th_relu = F.relu(hidden(th_relu))
        # Classification probabilities on k source domains.
        logprobs = []
        for i in range(self.num_domains):
            logprobs.append(F.log_softmax(self.softmax(sh_relu[i]), dim=1))
        # Domain classification accuracies.
        sdomains, tdomains = [], []
        for i in range(self.num_domains):
            sdomains.append(F.log_softmax(self.domains[i](self.grls[i](sh_relu[i])), dim=1))
            tdomains.append(F.log_softmax(self.domains[i](self.grls[i](th_relu)), dim=1))
        return logprobs, sdomains, tdomains

    def inference(self, inputs):
        h_relu = inputs
        for hidden in self.hiddens:
            h_relu = F.relu(hidden(h_relu))
        # Classification probability.
        logprobs = F.log_softmax(self.softmax(h_relu), dim=1)
        return logprobs

class MDANet_general(nn.Module):
    """
    Multi-layer perceptron with adversarial regularizer by domain classification.
    """
    def __init__(self, configs):
        super(MDANet_general, self).__init__()
        self.num_domains = configs["num_domains"]
        # Parameters of hidden, fully-connected layers, feature learning component.
        self.hiddens = configs['hiddens']
        # Parameter of the final softmax classification layer.
        self.predictor = configs['predictor']
        #Domain discriminator
        self.discriminator = configs['discriminator']
        # Gradient reversal layer.
        self.grls = [GradientReversalLayer() for _ in range(self.num_domains)]
        for i in range(configs['num_domains']):
            c=1
            for param in self.discriminator[i].parameters():
                self.register_parameter(name='disc'+str(i)+'-'+str(c), param=param)

    def forward(self, sinputs, tinputs):
        """
        :param sinputs:     A list of k inputs from k source domains.
        :param tinputs:     Input from the target domain.
        :return:
        """
        sh_relu, th_relu = [s.clone() for s in sinputs], tinputs.clone()
        for i in range(self.num_domains):
            for hidden in self.hiddens:
                sh_relu[i] = hidden(sh_relu[i])
        for hidden in self.hiddens:
            th_relu = hidden(th_relu)
        
        # Regressor
        vals = []
        for i in range(self.num_domains):
            sx = sh_relu[i].clone()
            for hiddens in self.predictor:
                sx = hiddens(sx)
            vals.append(sx)
            #logprobs.append(F.log_softmax(self.softmax(sh_relu[i]), dim=1))
         
        # Domain classification accuracies.
        sdomains, tdomains = [], []
        for i in range(self.num_domains):
            sx = sh_relu[i].clone()
            tx = th_relu.clone()
            for hiddens in self.discriminator[i]:
                sx = hiddens(sx)
                tx = hiddens(tx)
            sdomains.append(sx)
            tdomains.append(tx)
        return vals, sdomains, tdomains

    def inference(self, inputs):
        h_relu = inputs
        for hidden in self.hiddens:
            h_relu = hidden(h_relu)
        # Classification probability.
        for hidden in self.predictor:
            h_relu = hidden(h_relu)
        vals = h_relu
        return vals
    
    def extract_features(self, x):
        xx = x.clone()
        for hidden in self.hiddens:
            xx = hidden(xx)
        return xx