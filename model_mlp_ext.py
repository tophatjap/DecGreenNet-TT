#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from activations_ext import *



class nn_mlp_last_na_custom_act_w_param(nn.Module):
    
    def __init__(self, net_dims,  act_fun,   dropout = 0.0,):
        super(nn_mlp_last_na_custom_act_w_param, self).__init__()
        self.dropout = dropout
        self.act_fun = act_fun
        self.drop = nn.Dropout(p=dropout)
        self.net_dims = net_dims        
                    
        self.layers_X = nn.ModuleList()
        
        self.num_layers_X =  len(net_dims)-1 #.size(1)
        
       
        for i in range(self.num_layers_X):
            self.layers_X.append(nn.Linear(self.net_dims[i],self.net_dims[i+1]) )

    def forward(self,  input, param):
        out = input
        sx = torch.size(out)
        param_rep = param.repeat(sx[0],1)
        torch.cat((out ,param_rep),1)
        for i in range(self.num_layers_X-1):
            #print(i)
            out =self.drop(out)    
            out = self.layers_X[i](out) 
            #print(torch.norm(self.layers_X[i].weight))
            out = self.act_fun(out)
        out = self.layers_X[-1](out) 
            
        return out  
    
class nn_mlp_custom_act(nn.Module):
    
    def __init__(self, net_dims,  act_fun, dropout = 0.0 ):
        super(nn_mlp_custom_act, self).__init__()
        self.dropout = dropout
        self.act_fun = act_fun
        self.net_dims = net_dims
        self.drop = nn.Dropout(p=dropout)
            
        self.layers_X = nn.ModuleList()
        
        self.num_layers_X =  len(net_dims)-1 #.size(1)
      
        for i in range(self.num_layers_X):
            self.layers_X.append(nn.Linear(self.net_dims[i],self.net_dims[i+1]) )
       
    def forward(self,  input):
        out = input
        for i in range(self.num_layers_X):
            #print(i)
            out =self.drop(out)    
            out = self.layers_X[i](out) 
            #print(torch.norm(self.layers_X[i].weight))
            out = self.act_fun(out) #self.act(out)
            
        return out     


