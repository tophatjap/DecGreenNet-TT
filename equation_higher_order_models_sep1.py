#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class Poi2d_with_a(nn.Module):
    
    def __init__(self,device=0):
        super(Poi2d_with_a, self).__init__()
        self.device = device


    def forward(self,input,a):
        sx = input.size()
        
        out = torch.zeros([sx[0],1]).to(self.device)
        for i in range(sx[0]):
            out[i,0] =  -a*( input[i,0]**2 - input[i,0] + input[i,1]**2 - input[i,1]) 
           
        return out  

class Poi2d_with_a_4D(nn.Module):
    
    def __init__(self,device=0):
        super(Poi2d_with_a_4D, self).__init__()
        self.device = device


    def forward(self,input1,input2,param_a):
        
        a = input1[:,0]**2 - input1[:,0] + input1[:,1]**2 - input1[:,1]
        b = input2[:,0]**2 - input2[:,0] + input2[:,1]**2 - input2[:,1]
        sx = a.size()
        a = a.view(1,sx[0])
        b = b.view(1,sx[0])
        a_1 = torch.ones_like(a).view(1,sx[0])
        b_1 = torch.ones_like(b).view(1,sx[0])
        
        #print(a_1.size())
        #print(a.size())
        a_exp = a.t()@ a_1
        #print(a_exp)
        b_exp = (b.t()@ b_1).t()
        #print(b_exp)
        out = -param_a*(a_exp + b_exp) 
        #out[i,0] =  -a*( input[i,0]**2 - input[i,0] + input[i,1]**2 - input[i,1]) 
           
        return out  


class Poi2d_with_a_4D_1(nn.Module):
    
    def __init__(self,device=0):
        super(Poi2d_with_a_4D_1, self).__init__()
        self.device = device


    def forward(self,input1,input2,input3,input4,param_a):

        sx1 = input1.size()
        sx2 = input2.size()
        sx3 = input3.size()
        sx4 = input4.size()
        
        a = (input1[:,0]**2 - input1[:,0]).view(sx1[0],1) 
        b = (input2[:,0]**2 - input2[:,0]).view(sx1[0],1) 
        b = (input3[:,0]**2 - input3[:,0]).view(sx1[0],1) 
        d = (input4[:,0]**2 - input4[:,0]).view(sx1[0],1) 
        '''
        sx = a.size()
        a = a.view(1,sx[0])
        sx = b.size()
        b = b.view(1,sx[0])
        sx = c.size()
        c = c.view(1,sx[0])
        sx = d.size()
        d = d.view(1,sx[0])
        a_1 = torch.ones_like(a).view(1,sx[0])
        b_1 = torch.ones_like(b).view(1,sx[0])
        c_1 = torch.ones_like(c).view(1,sx[0])
        d_1 = torch.ones_like(d).view(1,sx[0])
        '''

        print(a)
        a_1 = a.repeat(1,sx2[0]*sx3[0]*sx4[0])
        a_1 = torch.reshape(a_1,(sx1[0],sx2[0],sx3[0],sx4[0]))
        
        b_1 = b.repeat(1,sx1[0]*sx3[0]*sx4[0])
        b_1 = torch.reshape(b_1,(sx2[0],sx3[0],sx4[0],sx1[0]))
        b_1 = torch.movedim(b_1,0,1)
        
        c_1 = c.repeat(1,sx1[0]*sx2[0]*sx4[0])
        c_1 = torch.reshape(c_1,(sx3[0],sx4[0],sx1[0],sx2[0]))
        c_1 = torch.movedim(c_1,0,1)
        
        d_1 = d.repeat(1,sx1[0]*sx2[0]*sx3[0])
        d_1 = torch.reshape(d_1,(sx4[0],sx1[0],sx2[0],sx3[0]))
        d_1 = torch.movedim(d_1,0,1)
        
        print(a_1)
        print(a_1[:,0,0,0])
        print(a_1[:,1,2,1])
        print(a_1.size())

        out = -param_a*(a_1 + b_1 + c_1 + d_1) 
        #out[i,0] =  -a*( input[i,0]**2 - input[i,0] + input[i,1]**2 - input[i,1]) 
           
        return out  


class Poi2d_with_a_exact(nn.Module):
    
    def __init__(self,device=0):
        super(Poi2d_with_a_exact, self).__init__()
        self.device = device


    def forward(self,input,a):
        sx = input.size()
        
        out = torch.zeros([sx[0],1]).to(self.device)
        for i in range(sx[0]):
            out[i,0] =  0.5*a*( input[i,0]*( input[i,0]-1)*(input[i,1]*(input[i,1]-1)) ) 
           
        return out  

##------------- Tensor N dim------------------------------------------------------------------------------------------------------------------

class Poi2d_with_a_tensor(nn.Module):
    
    def __init__(self,tensor_dim=2,device=0):
        super(Poi2d_with_a_tensor, self).__init__()
        self.device = device
        self.tensor_dim = tensor_dim
        self.num_dim = len(self.tensor_dim)
        self.dim_index = np.arange(self.num_dim)

    def forward(self,input,param_a):

        dim_shuff = self.tensor_dim  
        print(dim_shuff)
        sum_tensor = (input[0][:,0]**2 - input[0][:,0]).view(dim_shuff[0],1) 
        sum_tensor = sum_tensor.repeat(1, np.prod(dim_shuff[1:]))
        sum_tensor = torch.reshape(sum_tensor, dim_shuff)
        
        for i in range(self.num_dim-1):
            dim_shuff1 =list(np.roll(dim_shuff,-(i+1)))
            print(dim_shuff)
            a = (input[i+1][:,0]**2 - input[i+1][:,0]).view(dim_shuff1[0],1)
            a = a.repeat(1,np.prod(dim_shuff1[1:]))
            a = torch.reshape(a,dim_shuff1)
            print(a.shape)
            #a = torch.movedim(a,0,1)
            print(list(np.roll(self.dim_index,(i+1))))
            a = torch.permute(a, list(np.roll(self.dim_index,(i+1))))
            sum_tensor = sum_tensor + a
            
            

        '''
        sx1 = input1.size()
        sx2 = input2.size()
        sx3 = input3.size()
        sx4 = input4.size()
        
        a = (input1[:,0]**2 - input1[:,0]).view(sx1[0],1) 
        b = (input2[:,0]**2 - input2[:,0]).view(sx1[0],1) 
        b = (input3[:,0]**2 - input3[:,0]).view(sx1[0],1) 
        d = (input4[:,0]**2 - input4[:,0]).view(sx1[0],1) 
        '''
        '''
        sx = a.size()
        a = a.view(1,sx[0])
        sx = b.size()
        b = b.view(1,sx[0])
        sx = c.size()
        c = c.view(1,sx[0])
        sx = d.size()
        d = d.view(1,sx[0])
        a_1 = torch.ones_like(a).view(1,sx[0])
        b_1 = torch.ones_like(b).view(1,sx[0])
        c_1 = torch.ones_like(c).view(1,sx[0])
        d_1 = torch.ones_like(d).view(1,sx[0])
        '''
        
        '''
        print(a)
        a_1 = a.repeat(1,sx2[0]*sx3[0]*sx4[0])
        a_1 = torch.reshape(a_1,(sx1[0],sx2[0],sx3[0],sx4[0]))
        
        b_1 = b.repeat(1,sx1[0]*sx3[0]*sx4[0])
        b_1 = torch.reshape(b_1,(sx2[0],sx3[0],sx4[0],sx1[0]))
        b_1 = torch.movedim(b_1,0,1)
        
        c_1 = c.repeat(1,sx1[0]*sx2[0]*sx4[0])
        c_1 = torch.reshape(c_1,(sx3[0],sx4[0],sx1[0],sx2[0]))
        c_1 = torch.movedim(c_1,0,1)
        
        d_1 = d.repeat(1,sx1[0]*sx2[0]*sx3[0])
        d_1 = torch.reshape(d_1,(sx4[0],sx1[0],sx2[0],sx3[0]))
        d_1 = torch.movedim(d_1,0,1)
        
        print(a_1)
        print(a_1[:,0,0,0])
        print(a_1[:,1,2,1])
        print(a_1.size())
        
        out = -param_a*(a_1 + b_1 + c_1 + d_1) 
        #out[i,0] =  -a*( input[i,0]**2 - input[i,0] + input[i,1]**2 - input[i,1]) 
        '''
        
        out = -param_a*sum_tensor
        return out  

class Poi2d_with_a_tensor_cat(nn.Module):
    
    def __init__(self,tensor_dim=2,quad_concat=1,device=0):
        super(Poi2d_with_a_tensor_cat, self).__init__()
        self.device = device
        self.tensor_dim = tensor_dim
        self.num_dim = len(self.tensor_dim)
        self.dim_index = np.arange(self.num_dim)
        self.quad_concat = quad_concat
        self.num_quad = len(quad_concat)

    def forward(self,quad_x,param_a):

        dim_shuff = self.tensor_dim  
        #print(dim_shuff)
        idx = self.quad_concat[0]
        #print(quad_x[0].shape)
        #print(dim_shuff[0])
        a = (quad_x[idx[0]][:,0]**2 - quad_x[idx[0]][:,0]).view(dim_shuff[0],1)
        #print(q_x)
        for j in range(len(idx)-1):
            a = a + (quad_x[idx[j+1]][:,0]**2 - quad_x[idx[j+1]][:,0]).view(dim_shuff[0],1)
        a = a.repeat(1, np.prod(dim_shuff[1:]))
        sum_tensor = torch.reshape(a, dim_shuff)
        
        
        
        for i in range(self.num_quad-1):
            dim_shuff1 =list(np.roll(dim_shuff,-(i+1)))
            #print(dim_shuff)
            idx = self.quad_concat[i+1]
            #print(idx)
            a = (quad_x[idx[0]][:,0]**2 - quad_x[idx[0]][:,0]).view(dim_shuff1[0],1)
            #print(q_x)
            for j in range(len(idx)-1):
                a = a + (quad_x[idx[j+1]][:,0]**2 - quad_x[idx[j+1]][:,0]).view(dim_shuff1[0],1)
            a = a.repeat(1, np.prod(dim_shuff1[1:]))
            a = torch.reshape(a,dim_shuff1)
            #print(a.shape)
            #a = torch.movedim(a,0,1)
            #print(list(np.roll(self.dim_index,(i+1))))
            a = torch.permute(a, list(np.roll(self.dim_index,(i+1))))
            sum_tensor = sum_tensor + a
            
          
        
        out = -param_a*sum_tensor
        return out  

class Poi2d_with_a_sep(nn.Module):
    
    def __init__(self,device=0):
        super(Poi2d_with_a_sep, self).__init__()
        self.device = device
        

    def forward(self,quad_x,param_a):

        out1 = quad_x[0]**2 - quad_x[0]
        
        out2 = quad_x[1]**2 - quad_x[1]
    
        sx1 = out1.size()[0]
        sx2 = out2.size()[0]
        #print(sx1)
        #print(sx2)
        out1 =  out1.repeat(1,sx2)
        out2 =  out2.repeat(1,sx1)
        #print(out1.size())
        #print(out2.size())
        out = -param_a*(out1 + out2.T)


        return out , out 

##--- RF ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class RF_tensor_cat(nn.Module):
    
    def __init__(self,device=0):
        super(RF_tensor_cat, self).__init__()
        self.device = device

    def forward(self,quad_x,param_a):
        qx1 = quad_x[0]
        qx2 = quad_x[1]
        
        sx1 = qx1.size()
        sx2 = qx2.size()
        #print(self.pi)
        out = torch.zeros([sx1[0],sx2[0]]).to(self.device)
        for i in range(sx1[0]):
            for j in range(sx2[0]):
                x = qx1[i,0]
                y = qx2[j,0]
                u = math.exp(-(x**2 + 2*(y**2) + 1) )
                du1 = -u*2*x 
                du2 = -u*4*y
                du11 = (u**2)*4*(x**2) - u*2  
                du22 = (u**2)*16*(y**2) - u*4 
                out[i,j] = -( (du11 + du22)*(1 + 2*y**2) + 4*y*(du2) ) + (1 + x**2)*u
                
            
        return out  

##---- Poisson sin multi ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class PoiXd_tensor_mult(nn.Module):
    
    def __init__(self,tensor_dim=2,quad_concat=1,device=0):
        super(Poi2d_with_a_tensor_cat, self).__init__()
        self.device = device
        self.tensor_dim = tensor_dim
        self.num_dim = len(self.tensor_dim)
        self.dim_index = np.arange(self.num_dim)
        self.quad_concat = quad_concat
        self.num_quad = len(quad_concat)
        self.f_list = len()

    def forward(self,quad_x,param_a):

        dim_shuff = self.tensor_dim  
        #print(dim_shuff)
        idx = self.quad_concat[0]
        l1 =     torch.sin(torch.pi*quad_x[idx[0]][:,0])                       #(quad_x[idx[0]][:,0]**2 - quad_x[idx[0]][:,0]).view(dim_shuff[0],1)
        for j in range(len(idx)-1):
            l1 = l1*torch.sin(torch.pi*quad_x[idx[j+1]][:,0])   # + (quad_x[idx[j+1]][:,0]**2 - quad_x[idx[j+1]][:,0]).view(dim_shuff[0],1)
        #self.f_list.append()
        
        idx = self.quad_concat[1]
        l2 =     torch.sin(torch.pi*quad_x[idx[0]][:,0])                       #(quad_x[idx[0]][:,0]**2 - quad_x[idx[0]][:,0]).view(dim_shuff[0],1)
        for j in range(len(idx)-1):
            l2 = l2*torch.sin(torch.pi*quad_x[idx[j+1]][:,0])
        
        idx = self.quad_concat[2]
        l3 =     torch.sin(torch.pi*quad_x[idx[0]][:,0])                       #(quad_x[idx[0]][:,0]**2 - quad_x[idx[0]][:,0]).view(dim_shuff[0],1)
        for j in range(len(idx)-1):
            l3 = l3*torch.sin(torch.pi*quad_x[idx[j+1]][:,0])
        
            
        '''
        dim_shuff = self.tensor_dim  
        #print(dim_shuff)
        idx = self.quad_concat[0]
        #print(quad_x[0].shape)
        #print(dim_shuff[0])
        a = (quad_x[idx[0]][:,0]**2 - quad_x[idx[0]][:,0]).view(dim_shuff[0],1)
        #print(q_x)
        for j in range(len(idx)-1):
            a = a + (quad_x[idx[j+1]][:,0]**2 - quad_x[idx[j+1]][:,0]).view(dim_shuff[0],1)
        a = a.repeat(1, np.prod(dim_shuff[1:]))
        sum_tensor = torch.reshape(a, dim_shuff)
        
        
        
        for i in range(self.num_quad-1):
            dim_shuff1 =list(np.roll(dim_shuff,-(i+1)))
            #print(dim_shuff)
            idx = self.quad_concat[i+1]
            #print(idx)
            a = (quad_x[idx[0]][:,0]**2 - quad_x[idx[0]][:,0]).view(dim_shuff1[0],1)
            #print(q_x)
            for j in range(len(idx)-1):
                a = a + (quad_x[idx[j+1]][:,0]**2 - quad_x[idx[j+1]][:,0]).view(dim_shuff1[0],1)
            a = a.repeat(1, np.prod(dim_shuff1[1:]))
            a = torch.reshape(a,dim_shuff1)
            #print(a.shape)
            #a = torch.movedim(a,0,1)
            #print(list(np.roll(self.dim_index,(i+1))))
            a = torch.permute(a, list(np.roll(self.dim_index,(i+1))))
            sum_tensor = sum_tensor + a
            
          
        
        out = -param_a*sum_tensor
        '''
        
        
        return l1,l2,l3  
    
## poisson homogenous ----------------------------------------------------------------------------------------------


class Poi2d_homogen(nn.Module):
    
    def __init__(self, device=0):
        super(Poi2d_homogen, self).__init__()
        self.device = device


    def forward(self,input,a=1):
        sx = input.size()
        #out  = -1*sx[0]*(4*3.17**2)*torch.prod(torch.sin(2*3.17*input),1)
        out  = -1*sx[0]*(4*torch.pi**2)*torch.prod(torch.sin(2*torch.pi*input),1)
        #out = torch.zeros([sx[0],1]).to(self.device)
        #for i in range(sx[0]):
        #    out[i,0] =  -self.a*( input[i,0]**2 - input[i,0] + input[i,1]**2 - input[i,1]) 
           
        return out 
    
class PoiXd_homogen_tensor_cat(nn.Module):
    
    def __init__(self,tensor_dim=2,quad_concat=1,device=0):
        super(PoiXd_homogen_tensor_cat, self).__init__()
        self.device = device
        self.tensor_dim = tensor_dim
        self.num_dim = len(self.tensor_dim)
        self.dim_index = np.arange(self.num_dim)
        self.quad_concat = quad_concat
        self.num_quad = len(quad_concat)

    def forward(self,quad_x,param_a):
        dim = 0
        dim_shuff = self.tensor_dim  
        #print(dim_shuff)
        idx = self.quad_concat[0]
        #print(quad_x[0].shape)
        #print(dim_shuff[0])
        a = torch.sin(2*torch.pi*quad_x[idx[0]][:,0])
        dim = dim +1
        #print(q_x)
        for j in range(len(idx)-1):
            dim = dim +1
            a = a*torch.sin(2*torch.pi*quad_x[idx[j+1]][:,0])  #+ (quad_x[idx[j+1]][:,0]**2 - quad_x[idx[j+1]][:,0]).view(dim_shuff[0],1)
        a = a.repeat(1, np.prod(dim_shuff[1:]))
        prod_tensor = torch.reshape(a, dim_shuff)
        
        
        
        for i in range(self.num_quad-1):
            dim_shuff1 =list(np.roll(dim_shuff,-(i+1)))
            #print(dim_shuff)
            idx = self.quad_concat[i+1]
            #print(idx)
            
            a = torch.sin(2*torch.pi*quad_x[idx[0]][:,0])
            #print(q_x)
            dim = dim +1
            for j in range(len(idx)-1):
                dim = dim +1
                a = a*torch.sin(2*torch.pi*quad_x[idx[j+1]][:,0])
                
            a = a.repeat(1, np.prod(dim_shuff1[1:]))
            a = torch.reshape(a,dim_shuff1)
            #print(a.shape)
            #a = torch.movedim(a,0,1)
            #print(list(np.roll(self.dim_index,(i+1))))
            a = torch.permute(a, list(np.roll(self.dim_index,(i+1))))
            prod_tensor = prod_tensor*a
            
          
        #print(dim)
        out = -1*dim*(4*torch.pi**2)*prod_tensor
        return out  


class PoiXd_homogen_tensor_sep(nn.Module):
    
    def __init__(self,quad_concat=1,device=0):
        super(PoiXd_homogen_tensor_sep, self).__init__()
        self.device = device
        self.num_quad = len(quad_concat)
        #self.dim_index = np.arange(self.num_dim)
        self.quad_concat = quad_concat
        #self.list_sep_f = list()

    def forward(self,quad_x,param_a):
        dim = 0
        list_sep_f = list()
        idx = self.quad_concat[0]
        #print(quad_x[0].shape)
        #print(dim_shuff[0])
        a = torch.sin(2*torch.pi*quad_x[idx[0]][:,0])
        #a = torch.sin(2*quad_x[idx[0]][:,0])
        dim = dim +1
        #print(q_x)
        for j in range(len(idx)-1):
            dim = dim +1
            #a = a*torch.sin(2*torch.pi*quad_x[idx[j+1]][:,0])  #+ (quad_x[idx[j+1]][:,0]**2 - quad_x[idx[j+1]][:,0]).view(dim_shuff[0],1)
            #print(a)
            #print(a.size())
            #print(torch.sin(2*quad_x[idx[j+1]][:,0]))
            a = a*torch.sin(2*torch.pi*quad_x[idx[j+1]][:,0])
        #a = a.repeat(1, np.prod(dim_shuff[1:]))
        #prod_tensor = torch.reshape(a, dim_shuff)
        a = torch.reshape(a,(a.size()[0],1))
        #print(a)
        #print(a.size())
        list_sep_f.append(a)
        
        
        for i in range(self.num_quad-1):
            idx = self.quad_concat[i+1]
            #print(idx)
            
            #a = torch.sin(2*torch.pi*quad_x[idx[0]][:,0])
            a = torch.sin(2*torch.pi*quad_x[idx[0]][:,0])
            #print(q_x)
            dim = dim +1
            for j in range(len(idx)-1):
                dim = dim +1
                #a = a*torch.sin(2*torch.pi*quad_x[idx[j+1]][:,0])
                #print(a)
                #print(a.size())  
                a = a*torch.sin(2*torch.pi*quad_x[idx[j+1]][:,0])
            a = torch.reshape(a,(a.size()[0],1))    
            #print(a)
            #print(a.size())                
            list_sep_f.append(a)
            
          
        #print(dim)
        mult_coeff = -1*param_a*4*torch.pi**2
        return list_sep_f,   mult_coeff

class PoiXd_homogen(nn.Module):
    
    def __init__(self, device=0):
        super(PoiXd_homogen, self).__init__()
        self.device = device


    def forward(self,input,a=1):
        sx = input.size()
        pi = torch.acos(torch.zeros(1)).item()*2
        #out  = -1*sx[0]*(4*3.17**2)*torch.prod(torch.sin(2*3.17*input),1)
        out  = -1*sx[0]*(4*pi**2)*torch.prod(torch.sin(2*pi*input),1)
        #print(out.size())
        #out = torch.zeros([sx[0],1]).to(self.device)
        #for i in range(sx[0]):
        #    out[i,0] =  -self.a*( input[i,0]**2 - input[i,0] + input[i,1]**2 - input[i,1]) 
           
        return out  




##-- Poissson from MOD-NEt -------------------------------------------------------------------------------------------------------------------------------------------------

class PoiXd_a_tensor_sep(nn.Module):
    
    def __init__(self,num_dims=2,device=0):
        super(PoiXd_a_tensor_sep, self).__init__()
        self.device = device
        self.num_dims = num_dims
        #self.list_sep_f = list()

    def forward(self,quad_x,param_a):
        list_sep_f = list()
        for i in range(self.num_dims):
            a = quad_x[i][:,0]*(quad_x[i][:,0] -1)
            list_sep_f.append(a)
   
     
        mult_coeff = -param_a
        return list_sep_f,   mult_coeff
    

###-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Solution
##--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class PoiXd_homogen_sol(nn.Module):
    
    def __init__(self, device=0):
        super(PoiXd_homogen_sol, self).__init__()
        self.device = device


    def forward(self,input,a=1):
        sx = input.size()
        
        out  = torch.prod(torch.sin(2*np.pi*input),1)
        #out  = torch.prod(torch.sin(2*torch.pi*input),1)
        #print(out.size())
        #out = torch.zeros([sx[0],1]).to(self.device)
        #for i in range(sx[0]):
        #    out[i,0] =  -self.a*( input[i,0]**2 - input[i,0] + input[i,1]**2 - input[i,1]) 
           
        return out 