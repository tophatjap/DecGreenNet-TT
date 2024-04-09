#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from activations_ext import *
from model_mlp import *



class DecGreenNet_product_CP2(nn.Module):
    
    def __init__(self, mlp_x,  r_func ,mlp_quad,  mlp_quad_last_shape, quad_concat,   dropout = 0.0): #num_layers, 
        super(DecGreenNet_product_CP2, self).__init__()
        self.dropout = dropout
        self.mlp_x = mlp_x
        self.mlp_quad = mlp_quad
        self.mlp_quad_last_shape = mlp_quad_last_shape
        self.r_func = r_func 
        self.drop = nn.Dropout(p=dropout)
        self.quad_concat = quad_concat
        self.num_quad = len(quad_concat)
        

        self.einsum_chars = ('abcdefghijklmn')[0:self.num_quad+1]
        
        
    def forward(self,input, eq_param, quad_x):
        
        y, m = self.r_func(quad_x, eq_param)
        lhs = self.mlp_x(input)
        
        out_quad0 = self.mlp_quad[0](quad_x[0])
        out_quad1 = self.mlp_quad[1](quad_x[1])
        
        out_quad0 = y[0]*out_quad0
        out_quad1 = y[1]*out_quad1
        
        out_quad0 = torch.reshape(out_quad0, (out_quad0.size()[0] ,self.mlp_quad_last_shape[0],self.mlp_quad_last_shape[1]))
        out_quad1 = torch.reshape(out_quad1, (out_quad1.size()[0] ,self.mlp_quad_last_shape[0],self.mlp_quad_last_shape[1]))
        

        
        out_quad = torch.einsum('abx, cdx-> abcd'  ,out_quad0, out_quad1  )
        
        #out_quad = torch.sum(out_quad,(0,2,4))
        #lhs = torch.reshape(out_quad, (self.mlp_quad_last_shape[0]**3,1))
        
        out_quad = torch.movedim(out_quad, 2,1)  
        #print(out_quad.size())
        #out_quad = torch.movedim(out_quad, 4,2)
        sx = out_quad.size()
        out_quad = torch.reshape(out_quad,(sx[0]*sx[1], sx[2]*sx[3] ))    
        #sx = y.size()

        #y = torch.reshape(y, ( torch.prod(sx),1 ))
        
        #rhs = rhs*y
        rhs = torch.sum(out_quad,0)
        
        out = lhs@rhs.T
     
            
        return out
    
##-----------------------------------------------------------------------------------------

class DecGreenNet_product_CP2_non_sep(nn.Module):
    
    def __init__(self, mlp_x,  r_func ,mlp_quad,  mlp_quad_last_shape, quad_concat,   dropout = 0.0): #num_layers, 
        super(DecGreenNet_product_CP2_non_sep, self).__init__()
        self.dropout = dropout
        self.mlp_x = mlp_x
        self.mlp_quad = mlp_quad
        self.mlp_quad_last_shape = mlp_quad_last_shape
        self.r_func = r_func 
        self.drop = nn.Dropout(p=dropout)
        self.quad_concat = quad_concat
        self.num_quad = len(quad_concat)
        

        self.einsum_chars = ('abcdefghijklmn')[0:self.num_quad+1]
        
        
    def forward(self,input, eq_param, quad_x):
        
        y, m = self.r_func(quad_x, eq_param)
        lhs = self.mlp_x(input)
        
        out_quad0 = self.mlp_quad[0](quad_x[0])
        out_quad1 = self.mlp_quad[1](quad_x[1])
        
        out_quad0 = out_quad0
        out_quad1 = out_quad1
        
        out_quad0 = torch.reshape(out_quad0, (out_quad0.size()[0] ,self.mlp_quad_last_shape[0],self.mlp_quad_last_shape[1]))
        out_quad1 = torch.reshape(out_quad1, (out_quad1.size()[0] ,self.mlp_quad_last_shape[0],self.mlp_quad_last_shape[1]))
        

        
        out_quad = torch.einsum('abx, cdx-> abcd'  ,out_quad0, out_quad1  )
        
        #out_quad = torch.sum(out_quad,(0,2,4))
        #lhs = torch.reshape(out_quad, (self.mlp_quad_last_shape[0]**3,1))
        
        out_quad = torch.movedim(out_quad, 2,1)  
        #print(out_quad.size())
        #out_quad = torch.movedim(out_quad, 4,2)
        sx = out_quad.size()
        out_quad = torch.reshape(out_quad,(sx[0]*sx[1], sx[2]*sx[3] ))
        #print(sx[2]*sx[3])
        #print(sx[0]*sx[1])
        y = y.view(sx[0]*sx[1],1)
        #y = y.repeat(sx[0]*sx[1],1)
        out_quad = y*out_quad
        #out_quad = torch.reshape(out_quad,(sx[0]*sx[1], sx[2]*sx[3] ))
        #sx = y.size()

        #y = torch.reshape(y, ( torch.prod(sx),1 ))
        
        #rhs = rhs*y
        rhs = torch.sum(out_quad,0)
        
        out = lhs@rhs.T
     
            
        return out
    
## -------------------------------------------------------------------------------------------------

class DecGreenNet_product_CP3(nn.Module):
    
    def __init__(self, mlp_x,  r_func ,mlp_quad,  mlp_quad_last_shape, quad_concat,   dropout = 0.0): #num_layers, 
        super(DecGreenNet_product_CP3, self).__init__()
        self.dropout = dropout
        self.mlp_x = mlp_x
        self.mlp_quad = mlp_quad
        self.mlp_quad_last_shape = mlp_quad_last_shape
        self.r_func = r_func 
        self.drop = nn.Dropout(p=dropout)
        self.quad_concat = quad_concat
        self.num_quad = len(quad_concat)
        

        self.einsum_chars = ('abcdefghijklmn')[0:self.num_quad+1]
        
        
    def forward(self,input, eq_param, quad_x):
        
        y, m = self.r_func(quad_x, eq_param)
        lhs = self.mlp_x(input)
        
        out_quad0 = self.mlp_quad[0](quad_x[0])
        out_quad1 = self.mlp_quad[1](quad_x[1])
        out_quad2 = self.mlp_quad[2](quad_x[2])
        
        out_quad0 = y[0]*out_quad0
        out_quad1 = y[1]*out_quad1
        out_quad2 = y[2]*out_quad2
        
        out_quad0 = torch.reshape(out_quad0, (out_quad0.size()[0] ,self.mlp_quad_last_shape[0],self.mlp_quad_last_shape[1]))
        out_quad1 = torch.reshape(out_quad1, (out_quad1.size()[0] ,self.mlp_quad_last_shape[0],self.mlp_quad_last_shape[1]))
        out_quad2 = torch.reshape(out_quad2, (out_quad2.size()[0] ,self.mlp_quad_last_shape[0],self.mlp_quad_last_shape[1]))
        

        
        out_quad = torch.einsum('abx, cdx, efx -> abcdef'  ,out_quad0, out_quad1 , out_quad2 )
        
        #out_quad = torch.sum(out_quad,(0,2,4))
        #lhs = torch.reshape(out_quad, (self.mlp_quad_last_shape[0]**3,1))
        
        out_quad = torch.movedim(out_quad, 2,1)  
        #print(out_quad.size())
        out_quad = torch.movedim(out_quad, 4,2)
        sx = out_quad.size()
        out_quad = torch.reshape(out_quad,(sx[0]*sx[1]*sx[2], sx[3]*sx[4]*sx[5] ))    
        #sx = y.size()

        #y = torch.reshape(y, ( torch.prod(sx),1 ))
        
        #rhs = rhs*y
        rhs = torch.sum(out_quad,0)
        
        out = lhs@rhs.T
     
            
        return out
            
class DecGreenNet_product_CP4(nn.Module):
    
    def __init__(self, mlp_x,  r_func ,mlp_quad,  mlp_quad_last_shape, quad_concat,   dropout = 0.0): #num_layers, 
        super(DecGreenNet_product_CP4, self).__init__()
        self.dropout = dropout
        self.mlp_x = mlp_x
        self.mlp_quad = mlp_quad
        self.mlp_quad_last_shape = mlp_quad_last_shape
        self.r_func = r_func 
        self.drop = nn.Dropout(p=dropout)
        self.quad_concat = quad_concat
        self.num_quad = len(quad_concat)
        

        self.einsum_chars = ('abcdefghijklmn')[0:self.num_quad+1]
        
        
    def forward(self,input, eq_param, quad_x):
        
        y, m = self.r_func(quad_x, eq_param)
        lhs = self.mlp_x(input)
        
        out_quad0 = self.mlp_quad[0](quad_x[0])
        out_quad1 = self.mlp_quad[1](quad_x[1])
        out_quad2 = self.mlp_quad[2](quad_x[2])
        out_quad3 = self.mlp_quad[3](quad_x[3])
        
        out_quad0 = y[0]*out_quad0
        out_quad1 = y[1]*out_quad1
        out_quad2 = y[2]*out_quad2
        out_quad3 = y[3]*out_quad3
        
        out_quad0 = torch.reshape(out_quad0, (out_quad0.size()[0] ,self.mlp_quad_last_shape[0],self.mlp_quad_last_shape[1]))
        out_quad1 = torch.reshape(out_quad1, (out_quad1.size()[0] ,self.mlp_quad_last_shape[0],self.mlp_quad_last_shape[1]))
        out_quad2 = torch.reshape(out_quad2, (out_quad2.size()[0] ,self.mlp_quad_last_shape[0],self.mlp_quad_last_shape[1]))
        out_quad3 = torch.reshape(out_quad3, (out_quad3.size()[0] ,self.mlp_quad_last_shape[0],self.mlp_quad_last_shape[1]))        

        
        out_quad = torch.einsum('abx, cdx, efx, lmx -> abcdeflm'  ,out_quad0, out_quad1 , out_quad2 , out_quad3)
        
        #out_quad = torch.sum(out_quad,(0,2,4))
        #lhs = torch.reshape(out_quad, (self.mlp_quad_last_shape[0]**3,1))
        
        out_quad = torch.movedim(out_quad, 2,1)  
        #print(out_quad.size())
        out_quad = torch.movedim(out_quad, 4,2)
        out_quad = torch.movedim(out_quad, 6,3)
        sx = out_quad.size()
        out_quad = torch.reshape(out_quad,(sx[0]*sx[1]*sx[2]*sx[3], sx[4]*sx[5]*sx[6]*sx[7] ))    
        #sx = y.size()

        #y = torch.reshape(y, ( torch.prod(sx),1 ))
        
        #rhs = rhs*y
        rhs = torch.sum(out_quad,0)
        
        out = lhs@rhs.T
     
            
        return out
    
class DecGreenNet_product_CP5(nn.Module):
    
    def __init__(self, mlp_x,  r_func ,mlp_quad,  mlp_quad_last_shape, quad_concat,   dropout = 0.0): #num_layers, 
        super(DecGreenNet_product_CP5, self).__init__()
        self.dropout = dropout
        self.mlp_x = mlp_x
        self.mlp_quad = mlp_quad
        self.mlp_quad_last_shape = mlp_quad_last_shape
        self.r_func = r_func 
        self.drop = nn.Dropout(p=dropout)
        self.quad_concat = quad_concat
        self.num_quad = len(quad_concat)
        

        self.einsum_chars = ('abcdefghijklmn')[0:self.num_quad+1]
        
        
    def forward(self,input, eq_param, quad_x):
        
        y, m = self.r_func(quad_x, eq_param)
        lhs = self.mlp_x(input)
        
        out_quad0 = self.mlp_quad[0](quad_x[0])
        out_quad1 = self.mlp_quad[1](quad_x[1])
        out_quad2 = self.mlp_quad[2](quad_x[2])
        out_quad3 = self.mlp_quad[3](quad_x[3])
        out_quad4 = self.mlp_quad[4](quad_x[4])
        
        out_quad0 = y[0]*out_quad0
        out_quad1 = y[1]*out_quad1
        out_quad2 = y[2]*out_quad2
        out_quad3 = y[3]*out_quad3
        out_quad4 = y[4]*out_quad4
        
        out_quad0 = torch.reshape(out_quad0, (out_quad0.size()[0] ,self.mlp_quad_last_shape[0],self.mlp_quad_last_shape[1]))
        out_quad1 = torch.reshape(out_quad1, (out_quad1.size()[0] ,self.mlp_quad_last_shape[0],self.mlp_quad_last_shape[1]))
        out_quad2 = torch.reshape(out_quad2, (out_quad2.size()[0] ,self.mlp_quad_last_shape[0],self.mlp_quad_last_shape[1]))
        out_quad3 = torch.reshape(out_quad3, (out_quad3.size()[0] ,self.mlp_quad_last_shape[0],self.mlp_quad_last_shape[1]))        
        out_quad4 = torch.reshape(out_quad4, (out_quad4.size()[0] ,self.mlp_quad_last_shape[0],self.mlp_quad_last_shape[1]))     
        
        out_quad = torch.einsum('abx, cdx, efx, lmx , wvx -> abcdeflmwv'  ,out_quad0, out_quad1 , out_quad2 , out_quad3 , out_quad4)
        
        #out_quad = torch.sum(out_quad,(0,2,4))
        #lhs = torch.reshape(out_quad, (self.mlp_quad_last_shape[0]**3,1))
        
        out_quad = torch.movedim(out_quad, 2,1)  
        #print(out_quad.size())
        out_quad = torch.movedim(out_quad, 4,2)
        out_quad = torch.movedim(out_quad, 6,3)
        out_quad = torch.movedim(out_quad, 8,4)
        sx = out_quad.size()
        #print(out_quad.size())
        out_quad = torch.reshape(out_quad,(sx[0]*sx[1]*sx[2]*sx[3]*sx[4] , sx[5]*sx[6]*sx[7]*sx[8]*sx[9] ))    
        #sx = y.size()
        #print(out_quad.size())
        #y = torch.reshape(y, ( torch.prod(sx),1 ))
        
        #rhs = rhs*y
        rhs = torch.sum(out_quad,0)
        
        out = lhs@rhs.T
     
            
        return out