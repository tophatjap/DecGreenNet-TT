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

class DecGreenNet_product_tucker2(nn.Module):
    
    def __init__(self, mlp_x,  r_func ,mlp_quad,  mlp_quad_last_shape, quad_concat, core_mlp,  dropout = 0.0): #num_layers, 
        super(DecGreenNet_product_tucker2, self).__init__()
        self.dropout = dropout
        self.mlp_x = mlp_x
        self.mlp_quad = mlp_quad
        self.mlp_quad_last_shape = mlp_quad_last_shape
        self.r_func = r_func 
        self.drop = nn.Dropout(p=dropout)
        self.quad_concat = quad_concat
        self.num_quad = len(quad_concat)
        
        self.mlp_core =  core_mlp
        #self.core_init = core_init
        #self.einsum_chars = ('abcdefghijklmn')[0:self.num_quad+1]
        
        
    def forward(self,input, eq_param, quad_x, core_init ):
        
        y, m = self.r_func(quad_x, eq_param)
        lhs = self.mlp_x(input)
        
        core_out = self.mlp_core(core_init)
        sx_core_out = core_out.size()
        core_out = torch.reshape(core_out, (self.mlp_quad_last_shape[0], self.mlp_quad_last_shape[1], self.mlp_quad_last_shape[2] ))   
        #print(core_out.size())
        
        for i in range(self.num_quad):
            core_out = torch.movedim(core_out,1,0)
            #print(core_out.size())
            out_quad = self.mlp_quad[i](quad_x[i])
            #print(out_quad.size())
            #print(y)
            #print(y[i])
            out_quad = y[i]*out_quad
            #einsum_str = self.einsum_chars + ',x' + self.einsum_chars[i+1] + '->' + self.einsum_chars[0:i] + 'x' + self.einsum_chars[(i+2):]
            #print(einsum_str)
            core_out = torch.einsum('x..., cx -> c...'  ,core_out, out_quad )
            #print(core_out.size())
            core_out = torch.sum(core_out,0)
            #core_out = torch.einsum(einsum_str ,core_out, out_quad )
            core_out = core_out.squeeze()
          
        '''    
        sx = core_out.size()
        core_out = torch.reshape(core_out, (sx[0], torch.prod(sx[1:]) ) )
        y = torch.reshape(y, (1, torch.prod(sx[1:] ) )
        '''
        #rhs = rhs*y
        
        out = lhs@core_out.T


        #out = out*y

        #out = torch.sum(out,1)
        
            
        return out
    
## ---------------------------------------------------------------------------------------------------------------------

class DecGreenNet_product_tucker2_non_sep(nn.Module):
    
    def __init__(self, mlp_x,  r_func ,mlp_quad,  mlp_quad_last_shape, quad_concat, core_mlp,  dropout = 0.0): #num_layers, 
        super(DecGreenNet_product_tucker2_non_sep, self).__init__()
        self.dropout = dropout
        self.mlp_x = mlp_x
        self.mlp_quad = mlp_quad
        self.mlp_quad_last_shape = mlp_quad_last_shape
        self.r_func = r_func 
        self.drop = nn.Dropout(p=dropout)
        self.quad_concat = quad_concat
        self.num_quad = len(quad_concat)
        
        self.mlp_core =  core_mlp
        #self.core_init = core_init
        #self.einsum_chars = ('abcdefghijklmn')[0:self.num_quad+1]
        
        
    def forward(self,input, eq_param, quad_x, core_init ):
        
        y, m = self.r_func(quad_x, eq_param)
        lhs = self.mlp_x(input)
        
        core_out = self.mlp_core(core_init)
        #sx_core_out = core_out.size()
        core_out = torch.reshape(core_out, (self.mlp_quad_last_shape[0], self.mlp_quad_last_shape[1], self.mlp_quad_last_shape[2] ))   
        #print(core_out.size())
        out_quad0 = self.mlp_quad[0](quad_x[0])
        out_quad1 = self.mlp_quad[1](quad_x[1])
        #core_out = 
        '''
        for i in range(self.num_quad):
            core_out = torch.movedim(core_out,1,0)
            #print(core_out.size())
            out_quad = self.mlp_quad[i](quad_x[i])
            #print(out_quad.size())
            #print(y)
            #print(y[i])
            out_quad = out_quad
            #einsum_str = self.einsum_chars + ',x' + self.einsum_chars[i+1] + '->' + self.einsum_chars[0:i] + 'x' + self.einsum_chars[(i+2):]
            #print(einsum_str)
            core_out = torch.einsum('x..., cx -> c...'  ,core_out, out_quad )
            #print(core_out.size())
            core_out = torch.sum(core_out,0)
            #core_out = torch.einsum(einsum_str ,core_out, out_quad )
            core_out = core_out.squeeze()
        '''  
        '''    
        sx = core_out.size()
        core_out = torch.reshape(core_out, (sx[0], torch.prod(sx[1:]) ) )
        y = torch.reshape(y, (1, torch.prod(sx[1:] ) )
        '''
        #rhs = rhs*y
        y = y.view(sx[0]*sx[1],1)
        core_out = y*core_out
        
        
        out = lhs@core_out.T


        #out = out*y

        #out = torch.sum(out,1)
        
            
        return out



## ---------------------------------------------------------------------------------------------------------------------    

class DecGreenNet_product_tucker3(nn.Module):
    
    def __init__(self, mlp_x,  r_func ,mlp_quad,  mlp_quad_last_shape, quad_concat, core_mlp,  dropout = 0.0): #num_layers, 
        super(DecGreenNet_product_tucker3, self).__init__()
        self.dropout = dropout
        self.mlp_x = mlp_x
        self.mlp_quad = mlp_quad
        self.mlp_quad_last_shape = mlp_quad_last_shape
        self.r_func = r_func 
        self.drop = nn.Dropout(p=dropout)
        self.quad_concat = quad_concat
        self.num_quad = len(quad_concat)
        
        self.mlp_core =  core_mlp
        #self.core_init = core_init
        #self.einsum_chars = ('abcdefghijklmn')[0:self.num_quad+1]
        
        
    def forward(self,input, eq_param, quad_x, core_init ):
        
        y, m = self.r_func(quad_x, eq_param)
        lhs = self.mlp_x(input)
        
        core_out = self.mlp_core(core_init)
        sx_core_out = core_out.size()
        core_out = torch.reshape(core_out, (self.mlp_quad_last_shape[0], self.mlp_quad_last_shape[1], self.mlp_quad_last_shape[2] ,self.mlp_quad_last_shape[3] ))   
        #print(core_out.size())
        
        for i in range(self.num_quad):
            core_out = torch.movedim(core_out,1,0)
            #print(core_out.size())
            out_quad = self.mlp_quad[i](quad_x[i])
            #print(out_quad.size())
            #print(y)
            #print(y[i])
            out_quad = y[i]*out_quad
            #einsum_str = self.einsum_chars + ',x' + self.einsum_chars[i+1] + '->' + self.einsum_chars[0:i] + 'x' + self.einsum_chars[(i+2):]
            #print(einsum_str)
            core_out = torch.einsum('x..., cx -> c...'  ,core_out, out_quad )
            #print(core_out.size())
            core_out = torch.sum(core_out,0)
            #core_out = torch.einsum(einsum_str ,core_out, out_quad )
            core_out = core_out.squeeze()
          
        '''    
        sx = core_out.size()
        core_out = torch.reshape(core_out, (sx[0], torch.prod(sx[1:]) ) )
        y = torch.reshape(y, (1, torch.prod(sx[1:] ) )
        '''
        #rhs = rhs*y
        
        out = lhs@core_out.T


        #out = out*y

        #out = torch.sum(out,1)
        
            
        return out
  
class DecGreenNet_product_tucker4(nn.Module):
    
    def __init__(self, mlp_x,  r_func ,mlp_quad,  mlp_quad_last_shape, quad_concat, core_mlp,  dropout = 0.0): #num_layers, 
        super(DecGreenNet_product_tucker4, self).__init__()
        self.dropout = dropout
        self.mlp_x = mlp_x
        self.mlp_quad = mlp_quad
        self.mlp_quad_last_shape = mlp_quad_last_shape
        self.r_func = r_func 
        self.drop = nn.Dropout(p=dropout)
        self.quad_concat = quad_concat
        self.num_quad = len(quad_concat)
        
        self.mlp_core =  core_mlp
        #self.core_init = core_init
        #self.einsum_chars = ('abcdefghijklmn')[0:self.num_quad+1]
        
        
    def forward(self,input, eq_param, quad_x, core_init ):
        
        y, m = self.r_func(quad_x, eq_param)
        lhs = self.mlp_x(input)
        
        core_out = self.mlp_core(core_init)
        sx_core_out = core_out.size()
        core_out = torch.reshape(core_out, (self.mlp_quad_last_shape[0], self.mlp_quad_last_shape[1], self.mlp_quad_last_shape[2] ,self.mlp_quad_last_shape[3], self.mlp_quad_last_shape[4] ))   
        #print(core_out.size())
        
        for i in range(self.num_quad):
            core_out = torch.movedim(core_out,1,0)
            #print(core_out.size())
            out_quad = self.mlp_quad[i](quad_x[i])
            #print(out_quad.size())
            #print(y)
            #print(y[i])
            out_quad = y[i]*out_quad
            #einsum_str = self.einsum_chars + ',x' + self.einsum_chars[i+1] + '->' + self.einsum_chars[0:i] + 'x' + self.einsum_chars[(i+2):]
            #print(einsum_str)
            core_out = torch.einsum('x..., cx -> c...'  ,core_out, out_quad )
            #print(core_out.size())
            core_out = torch.sum(core_out,0)
            #core_out = torch.einsum(einsum_str ,core_out, out_quad )
            core_out = core_out.squeeze()
          
        '''    
        sx = core_out.size()
        core_out = torch.reshape(core_out, (sx[0], torch.prod(sx[1:]) ) )
        y = torch.reshape(y, (1, torch.prod(sx[1:] ) )
        '''
        #rhs = rhs*y
        
        out = lhs@core_out.T


        #out = out*y

        #out = torch.sum(out,1)
        
            
        return out          

## -- paramter core ------------------------------------------------------------------------------------------------------------

class DecGreenNet_product_tucker_param_2(nn.Module):
    
    def __init__(self, mlp_x,  r_func ,mlp_quad,  mlp_quad_last_shape, quad_concat, mlp_dims,  dropout = 0.0): #num_layers, 
        super(DecGreenNet_product_tucker_param_2, self).__init__()
        self.dropout = dropout
        self.mlp_x = mlp_x
        self.mlp_quad = mlp_quad
        self.mlp_quad_last_shape = mlp_quad_last_shape
        self.r_func = r_func 
        self.drop = nn.Dropout(p=dropout)
        self.quad_concat = quad_concat
        self.num_quad = len(quad_concat)
        self.mlp_dims = mlp_dims
        self.mlp_core =  Parameter(torch.FloatTensor(self.mlp_dims[0], self.mlp_dims[1], self.mlp_dims[2] ))
        self.init_core()
        #self.core_init = core_init
        #self.einsum_chars = ('abcdefghijklmn')[0:self.num_quad+1]

    def init_core(self):
        stdv = 1. / math.sqrt(self.mlp_dims[0])
        self.mlp_core.data.uniform_(-stdv, stdv)        
        
    def forward(self,input, eq_param, quad_x ):
        
        y, m = self.r_func(quad_x, eq_param)
        lhs = self.mlp_x(input)
        
        core_out = self.mlp_core
        #print(self.mlp_core)
        for i in range(self.num_quad):
            core_out = torch.movedim(core_out,1,0)
            #print(core_out.size())
          
            out_quad = self.mlp_quad[i](quad_x[i])
            #print(out_quad.size())
            #print(y)
            #print(y[i])
            out_quad = y[i]*out_quad
            #einsum_str = self.einsum_chars + ',x' + self.einsum_chars[i+1] + '->' + self.einsum_chars[0:i] + 'x' + self.einsum_chars[(i+2):]
            #print(einsum_str)
            core_out = torch.einsum('x..., cx -> c...'  ,core_out, out_quad )
            #print(core_out.size())
            core_out = torch.sum(core_out,0)
            #core_out = torch.einsum(einsum_str ,core_out, out_quad )
            core_out = core_out.squeeze()
          
        '''    
        sx = core_out.size()
        core_out = torch.reshape(core_out, (sx[0], torch.prod(sx[1:]) ) )
        y = torch.reshape(y, (1, torch.prod(sx[1:] ) )
        '''
        #rhs = rhs*y
        
        out = lhs@core_out.T


        #out = out*y

        #out = torch.sum(out,1)
        
            
        return out

##---------------------------------------------------------------------------------------------------------

class DecGreenNet_product_tucker_param_2_non_sep(nn.Module):
    
    def __init__(self, mlp_x,  r_func ,mlp_quad,  mlp_quad_last_shape, quad_concat, mlp_dims,  dropout = 0.0): #num_layers, 
        super(DecGreenNet_product_tucker_param_2_non_sep, self).__init__()
        self.dropout = dropout
        self.mlp_x = mlp_x
        self.mlp_quad = mlp_quad
        self.mlp_quad_last_shape = mlp_quad_last_shape
        self.r_func = r_func 
        self.drop = nn.Dropout(p=dropout)
        self.quad_concat = quad_concat
        self.num_quad = len(quad_concat)
        self.mlp_dims = mlp_dims
        self.mlp_core =  Parameter(torch.FloatTensor(self.mlp_dims[0], self.mlp_dims[1], self.mlp_dims[2] ))
        self.init_core()
        #self.core_init = core_init
        #self.einsum_chars = ('abcdefghijklmn')[0:self.num_quad+1]

    def init_core(self):
        stdv = 1. / math.sqrt(self.mlp_dims[0])
        self.mlp_core.data.uniform_(-stdv, stdv)        
        
    def forward(self,input, eq_param, quad_x ):
        
        y, m = self.r_func(quad_x, eq_param)
        lhs = self.mlp_x(input)
        
        core_out = self.mlp_core
        #print(self.mlp_core)
        out_quad0 = self.mlp_quad[0](quad_x[0])
        out_quad1 = self.mlp_quad[1](quad_x[1])
        #print(out_quad0.size())
        #print(out_quad1.size())
        #print(core_out.size())
        core_out = torch.einsum('axy,  bx, cy -> abc' , core_out, out_quad0, out_quad1)
        '''
        for i in range(self.num_quad):
            core_out = torch.movedim(core_out,1,0)
            #print(core_out.size())
          
            out_quad = self.mlp_quad[i](quad_x[i])
            #print(out_quad.size())
            #print(y)
            #print(y[i])
            #out_quad = out_quad
            #print(core_out.size())
            #einsum_str = self.einsum_chars + ',x' + self.einsum_chars[i+1] + '->' + self.einsum_chars[0:i] + 'x' + self.einsum_chars[(i+2):]
            #print(einsum_str)
            core_out = torch.einsum('x..., cx -> c...'  ,core_out, out_quad )
            #print(core_out.size())
            core_out = torch.sum(core_out,0)
            #core_out = torch.einsum(einsum_str ,core_out, out_quad )
            core_out = core_out.squeeze()
        '''    
          
        '''    
        sx = core_out.size()
        core_out = torch.reshape(core_out, (sx[0], torch.prod(sx[1:]) ) )
        y = torch.reshape(y, (1, torch.prod(sx[1:] ) )
        '''
        #rhs = rhs*y
        sx = core_out.size()
        core_out = torch.reshape(core_out , (sx[0], sx[1]*sx[2] ) )
        sx = y.size()
        y = y.view(sx[0]*sx[1],1)
        core_out = y*core_out.T
        out = lhs@core_out.T
        #print(out.size())

        #out = out*y

        out = torch.sum(out,1)
        #print(out.size())
            
        return out


##---------------------------------------------------------------------------------------------------------

class DecGreenNet_product_tucker_param_3(nn.Module):
    
    def __init__(self, mlp_x,  r_func ,mlp_quad,  mlp_quad_last_shape, quad_concat, mlp_dims,  dropout = 0.0): #num_layers, 
        super(DecGreenNet_product_tucker_param_3, self).__init__()
        self.dropout = dropout
        self.mlp_x = mlp_x
        self.mlp_quad = mlp_quad
        self.mlp_quad_last_shape = mlp_quad_last_shape
        self.r_func = r_func 
        self.drop = nn.Dropout(p=dropout)
        self.quad_concat = quad_concat
        self.num_quad = len(quad_concat)
        self.mlp_dims = mlp_dims
        self.mlp_core =  Parameter(torch.FloatTensor(self.mlp_dims[0], self.mlp_dims[1], self.mlp_dims[2], self.mlp_dims[3]))
        self.init_core()
        #self.core_init = core_init
        #self.einsum_chars = ('abcdefghijklmn')[0:self.num_quad+1]

    def init_core(self):
        stdv = 1. / math.sqrt(self.mlp_dims[0])
        self.mlp_core.data.uniform_(-stdv, stdv)        
        
    def forward(self,input, eq_param, quad_x ):
        
        y, m = self.r_func(quad_x, eq_param)
        lhs = self.mlp_x(input)
        
        core_out = self.mlp_core
        #print(self.mlp_core)
        for i in range(self.num_quad):
            core_out = torch.movedim(core_out,1,0)
            #print(core_out.size())
          
            out_quad = self.mlp_quad[i](quad_x[i])
            #print(out_quad.size())
            #print(y)
            #print(y[i])
            out_quad = y[i]*out_quad
            #einsum_str = self.einsum_chars + ',x' + self.einsum_chars[i+1] + '->' + self.einsum_chars[0:i] + 'x' + self.einsum_chars[(i+2):]
            #print(einsum_str)
            core_out = torch.einsum('x..., cx -> c...'  ,core_out, out_quad )
            #print(core_out.size())
            core_out = torch.sum(core_out,0)
            #core_out = torch.einsum(einsum_str ,core_out, out_quad )
            core_out = core_out.squeeze()
          
        '''    
        sx = core_out.size()
        core_out = torch.reshape(core_out, (sx[0], torch.prod(sx[1:]) ) )
        y = torch.reshape(y, (1, torch.prod(sx[1:] ) )
        '''
        #rhs = rhs*y
        
        out = lhs@core_out.T


        #out = out*y

        #out = torch.sum(out,1)
        
            
        return out
  
class DecGreenNet_product_tucker_param_4(nn.Module):
    
    def __init__(self, mlp_x,  r_func ,mlp_quad,  mlp_quad_last_shape, quad_concat, mlp_dims,  dropout = 0.0): #num_layers, 
        super(DecGreenNet_product_tucker_param_4, self).__init__()
        self.dropout = dropout
        self.mlp_x = mlp_x
        self.mlp_quad = mlp_quad
        self.mlp_quad_last_shape = mlp_quad_last_shape
        self.r_func = r_func 
        self.drop = nn.Dropout(p=dropout)
        self.quad_concat = quad_concat
        self.num_quad = len(quad_concat)
        self.mlp_dims = mlp_dims
        self.mlp_core =  Parameter(torch.FloatTensor(self.mlp_dims[0], self.mlp_dims[1], self.mlp_dims[2], self.mlp_dims[3], self.mlp_dims[4]))
        self.init_core()
        #self.core_init = core_init
        #self.einsum_chars = ('abcdefghijklmn')[0:self.num_quad+1]

    def init_core(self):
        stdv = 1. / math.sqrt(self.mlp_dims[0])
        self.mlp_core.data.uniform_(-stdv, stdv)   
        
        
    def forward(self,input, eq_param, quad_x  ):
        
        y, m = self.r_func(quad_x, eq_param)
        lhs = self.mlp_x(input)
        
        
        core_out = self.mlp_core
        for i in range(self.num_quad):
            core_out = torch.movedim(core_out,1,0)
            #print(core_out.size())
            out_quad = self.mlp_quad[i](quad_x[i])
            #print(out_quad.size())
            #print(y)
            #print(y[i])
            out_quad = y[i]*out_quad
            #einsum_str = self.einsum_chars + ',x' + self.einsum_chars[i+1] + '->' + self.einsum_chars[0:i] + 'x' + self.einsum_chars[(i+2):]
            #print(einsum_str)
            core_out = torch.einsum('x..., cx -> c...'  ,core_out, out_quad )
            #print(core_out.size())
            core_out = torch.sum(core_out,0)
            #core_out = torch.einsum(einsum_str ,core_out, out_quad )
            core_out = core_out.squeeze()
          
        '''    
        sx = core_out.size()
        core_out = torch.reshape(core_out, (sx[0], torch.prod(sx[1:]) ) )
        y = torch.reshape(y, (1, torch.prod(sx[1:] ) )
        '''
        #rhs = rhs*y
        
        out = lhs@core_out.T


        #out = out*y

        #out = torch.sum(out,1)
        
            
        return out    

'''
class DecGreenNet_product_tucker_(nn.Module):
    
    def __init__(self, mlp_x,  r_func ,mlp_quad,  mlp_quad_last_shape, quad_concat, core_mlp,  dropout = 0.0): #num_layers, 
        super(DecGreenNet_product_tucker, self).__init__()
        self.dropout = dropout
        self.mlp_x = mlp_x
        self.mlp_quad = mlp_quad
        self.mlp_quad_last_shape = mlp_quad_last_shape
        self.r_func = r_func 
        self.drop = nn.Dropout(p=dropout)
        self.quad_concat = quad_concat
        self.num_quad = len(quad_concat)
        
        self.mlp_core =  core_mlp
        #self.core_init = core_init
        self.einsum_chars = ('abcdefghijklmn')[0:self.num_quad+1]
        
        
    def forward(self,input, eq_param, quad_x, core_init ):
        
        y = self.r_func(quad_x, eq_param)
        lhs = self.mlp_x(input)
        
        core_out = self.mlp_core(core_init)
          
        
        for i in range(self.num_quad):
            out_quad = self.mlp_quad[i](quad_x[i])
            einsum_str = self.einsum_chars + ',x' + self.einsum_chars[i+1] + '->' + self.einsum_chars[0:i] + 'x' + self.einsum_chars[(i+2):]
            core_out = torch.einsum(einsum_str ,core_out, out_quad )
            
        sx = core_out.size()
        core_out = torch.reshape(core_out, (sx[0], torch.prod(sx[1:]) ) )
        y = torch.reshape(y, (1, torch.prod(sx[1:] ) ))
        
        rhs = rhs*y
        
        out = lhs@rhs


        #out = out*y

        out = torch.sum(out,1)
        
            
        return out
'''            