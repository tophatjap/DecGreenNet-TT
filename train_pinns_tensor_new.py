#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 13:31:40 2022

@author: kishan
"""
import time
import random
import argparse
import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from model_pinns_sep import * 
from activations_ext import * 
from data_sampler import * 
import matplotlib.pyplot as plt
import scipy.io as sio
import uuid
import torch.optim as optim
from torch.autograd import Variable


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=5000, help='Number of epochs to train.')#1000
parser.add_argument('--lr', type=float, default=0.001, help='learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay (L2 loss on parameters).')
parser.add_argument('--layer', type=int, default=4, help='Number of layers.')
parser.add_argument('--hidden', type=int, default=64, help='hidden dimensions.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--data', default='poisson3D_homogenous_', help='dateset')
parser.add_argument('--num_quad', type=int, default=10, help='number of quadrature points')
parser.add_argument('--r', type=int, default=5, help='number of quadrature points')
parser.add_argument('--r1', type=int, default=5, help='number of quadrature points')
parser.add_argument('--dev', type=int, default=0, help='device id')
parser.add_argument('--act_type_x', type=str, default='relu3', help='activation on input x')
parser.add_argument('--act_type_quad', type=str, default='relu3', help='activation on input quadrature points')
parser.add_argument('--act_type_last', type=str, default='relu3', help='activation on output layer')
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--full', type=int, default=0)
parser.add_argument('--opt_method', type=str, default='Adam')
args = parser.parse_args()

batch_size = args.batch_size # 512
hidden = [64] 
lr = args.lr
wd = [0] #, 0.000001,0.001]
drop = [0.0] #, 0.1,0.5,0.7]
reg_para1 = [1]
reg_para2 = [1] #,0.01,0.0001] # 1e-2,1e-3,1e-4]
epoch = args.epochs
x_dim = 3
qx_dim = 1
r = args.r
r1 = args.r1
num_quad=args.num_quad
num_boundaries = 12
#t_rank = 10

#num_tensor_dim = 5
#tensor_dim = [2,2,2,2,2]
mlp_quad_last_shape = [[r,1],[r,r],[r,1]]
input_concat = [[0],[1],[2]]
tensor_quad_dim = [num_quad,num_quad,num_quad]
no_samples = 1




net_config_X1= list()
net_config_X1.append([qx_dim, 64,64,  r1])
net_config_X1.append([qx_dim, 16,16,  r1])
#net_config_X1.append([x_dim, 128,128,128,128,  r1])
#net_config_X1.append([x_dim, 256,256,256,256,  r1])
#net_config_X1.append([x_dim, 512,512,512,512,  r1])


net_config_X2 = list()
#net_config_outer_arr.append([qx_dim, 32,r*r])
#net_config_outer_arr.append([qx_dim, 32,32,32,r*r])
net_config_X2.append([qx_dim, 64,64,r*r])
net_config_X2.append([qx_dim, 64,64,r*r])
#net_config_X2.append([qx_dim, 128,128,r1*r])

net_config_X3 = list()
#net_config_outer_arr.append([qx_dim, 32,r*r])
#net_config_outer_arr.append([qx_dim, 32,32,32,r*r])
net_config_X3.append([qx_dim, 16,16,r])
net_config_X3.append([qx_dim, 64,64,r])
net_config_X3.append([qx_dim, 64,64,64,r])





fname = 'data_new/poisson_homogenous3_num_boundaries_12_D_no_boundary_.mat'

  

res_array = np.zeros([len(wd) ,len(drop) ,len(reg_para1)  ,len(reg_para2)  ,  no_samples])
cudaid =    "cuda:" + str(args.dev)  #
device = torch.device(cudaid)
checkpt_file = 'pretrained/' + uuid.uuid4().hex + '.pt'
print(cudaid, checkpt_file)

#samples
sampler = Data_Sampler(fname,batch_size=batch_size, num_b=12, device=device)
sampler.load_eq_param()
sampler.load_interior()
sampler.load_boundary()   


# initialize varaibles
var_train_data = list() 
var_test_data = list() 
var_val_data = list() 

for i in range(len(sampler.eq_param)):
    var_train_data.append(Variable(sampler.train_data_i[i] , requires_grad = True).to(device))
    var_test_data.append(Variable(sampler.test_data_i[i] , requires_grad = True).to(device))
    var_val_data.append(Variable(sampler.val_data_i[i] , requires_grad = True).to(device))

q_x = list()
quad_x = list() 
for p in range(x_dim):
    q_x1 = 2*(np.random.rand(num_quad,1) - 0.5)
    quad_x1 = torch.FloatTensor(q_x1).to(device)
    q_x.append(q_x1)
    quad_x.append(quad_x1)


for net_config_X1_ in net_config_X1:
    for net_config_X2_ in net_config_X2:
        for net_config_X3_ in net_config_X3:
            for wd_ in wd:
                for drop_ in drop:
                    for reg_p1 in reg_para1:
                        for reg_p2 in reg_para2:
                            #for reg_p3 in reg_para3:
            
                            random.seed(args.seed)
                            np.random.seed(args.seed)
                            torch.manual_seed(args.seed)
                            torch.cuda.manual_seed(args.seed)
                    
                            for j in range(no_samples):
                    
                                def train_step(model,sampler, optimizer):
                                    loss = nn.MSELoss()
                                    model.train()
                                    
                                    optimizer.zero_grad()
                                    if args.opt_method == 'Adam':
                                        if args.full == 0:
                                            x,y,rnd_s_list  = sampler.rnd_sample(data_trpe = 'train')
                                        else:
                                            x,y,rnd_s_list  = sampler.full_indexes(data_trpe = 'train')
                                    elif args.opt_method == 'LBFGS':
                                        x,y,rnd_s_list = sampler.full_indexes(data_trpe = 'train')
                                    
                                    loss_train = 0
                                    for i in range(len(sampler.eq_param)): #len(a)):
                                        #print(rnd_s_list[i])
                                        #print(var_train_data[i])
                                        d = model(var_train_data[i])
                                        
                                        d = torch.autograd.grad(d,var_train_data[i] , grad_outputs=torch.ones_like(d), create_graph=True )
                                        dx_total = 0
                                        for di in range(x_dim):
                                            dx1 = d[0][:,di].reshape(-1,1)
                                            dxx1 = torch.autograd.grad(dx1, var_train_data[i] , grad_outputs=torch.ones_like(dx1), create_graph=True )[0][:,di].reshape(-1,1)
                                            dx_total =  dx_total  +  dxx1[rnd_s_list[i]] 
                                            
                                        loss_train = loss_train + reg_p1*loss((dx_total).squeeze(), y[i].squeeze())
                                    loss_train = loss_train/len(sampler.eq_param)
                                    
                                    
                                    if args.opt_method == 'Adam':
                                        if args.full == 0:
                                            x_b,y_b,rnd_sample_b  = sampler.rnd_sample_b(data_trpe = 'train')
                                        else:
                                            x_b,y_b,rnd_sample_b  = sampler.rnd_full_indexes_b(data_trpe = 'train')
                                    elif args.opt_method == 'LBFGS':
                                        x_b,y_b,rnd_sample_b  = sampler.rnd_full_indexes_b(data_trpe = 'train')
                                    #x_b,y_b,rnd_sample_b = sampler.rnd_sample_b(data_trpe = 'train')
                                    for j in range(num_boundaries):
                                        temp = 0
                                        for i in range(len(sampler.eq_param)): #len(a)):
                                            #print(x_b[(i-1)*4+j])
                                            #print(x_b[(i-1)*4+j])
                                            #print((i-1)*num_boundaries+j)
                                            #print(x_b[(i-1)*num_boundaries+j])
                                            d = model(x_b[(i-1)*num_boundaries+j])
                                            #print(x_b[(i-1)*4+j].size())
                                            #print(d.squeeze().size())
                                            #print(y_b[(i-1)*4+j].squeeze().size())
                                            temp = temp + reg_p2*loss(d.squeeze(), y_b[(i-1)*num_boundaries+j].squeeze()) #+ reg_p3*loss(vx, vx_rhs) 
                                        #print(temp)
                                        loss_train = loss_train + temp/len(sampler.eq_param)
                                   
                                    
                                    #print("Train loss :" + str(loss_train.item()) )
                                    loss_train.backward()
                                    optimizer.step()
                                    return loss_train.item() #,acc_train.item()
                    
                                def test_step(model,sampler):
                                    model.eval()
                                   # with torch.no_grad():
                                    loss = nn.MSELoss()
                                    #model.eval()
                                    loss_test_ = list() 
                                    for k in range(len(sampler.eq_param)):
                                        if args.opt_method == 'Adam':
                                            if args.full == 0:
                                                x,y,rnd_s_list  = sampler.rnd_sample(data_trpe = 'test')
                                            else:
                                                x,y,rnd_s_list  = sampler.full_indexes(data_trpe = 'test')
                                        elif args.opt_method == 'LBFGS':
                                            x,y,rnd_s_list = sampler.full_indexes(data_trpe = 'test')
            
                                        loss_test = 0
                                        #loss_test_u = 0
                                        #loss_test_u_size = 0
                                        #for i in range(len(sampler.eq_param)): #len(a)):
                                        d_ = model(var_test_data[k])
                                        d = torch.autograd.grad(d_,var_test_data[k] , grad_outputs=torch.ones_like(d_), create_graph=True )
                                        
                                        dx_total = 0
                                        for di in range(x_dim):
                                            dx1 = d[0][:,di].reshape(-1,1)
                                            dxx1 = torch.autograd.grad(dx1, var_test_data[k] , grad_outputs=torch.ones_like(dx1), create_graph=True )[0][:,di].reshape(-1,1)
                                            dx_total =  dx_total  +  dxx1[rnd_s_list[k]] 
                                            
                                        loss_test = loss_test + reg_p1*loss((dx_total).squeeze(), y[k].squeeze())
                                        #loss_test_u = torch.sum((d_[rnd_s_list[k]] - y_u[0])**2)
                                        #loss_test_u_size = y_u[0].size(0)
                                        #print(loss_test_u_size)
                                        #print(loss_test_u)
                                       
                                         
                                        if args.opt_method == 'Adam':
                                            if args.full == 0:
                                                x_b,y_b,rnd_sample_b  = sampler.rnd_sample_b(data_trpe = 'test')
                                            else:
                                                x_b,y_b,rnd_sample_b  = sampler.rnd_full_indexes_b(data_trpe = 'test')
                                        elif args.opt_method == 'LBFGS':
                                            x_b,y_b,rnd_sample_b  = sampler.rnd_full_indexes_b(data_trpe = 'test')
            
                                        temp = 0
                                        for j in range(num_boundaries):
                                            d = model(x_b[(k-1)*num_boundaries+j])
                                            temp = temp + reg_p2*loss(d.squeeze(), y_b[(k-1)*num_boundaries+j].squeeze()) #+ reg_p3*loss(vx, vx_rhs) 
                                            loss_test = loss_test + temp
                                            loss_test_.append(loss_test.item())
                                            
                                            #loss_test_u = loss_test_u + torch.sum((d - y_b[(k-1)*num_boundaries+j])**2)
                                            #loss_test_u_size = loss_test_u_size + y_b[(k-1)*num_boundaries+j].size(0)
                                            #rint(loss_test_u_size)
                                            #print(loss_test_u)
                                        #loss_test_u = (loss_test_u/loss_test_u_size)**(0.5)
                                        #print(loss_test_u)
                                        print("data set " + str(k) + " test error " + str(loss_test.item()) )
                                        
                                    
                                    if args.opt_method == 'Adam':
                                        if args.full == 0:
                                            x,y,rnd_s_list  = sampler.rnd_sample(data_trpe = 'test')
                                        else:
                                            x,y,rnd_s_list  = sampler.full_indexes(data_trpe = 'test')
                                    elif args.opt_method == 'LBFGS':
                                        x,y,rnd_s_list = sampler.full_indexes(data_trpe = 'test')
        
                                    loss_test = 0
                                    #loss_test_u1 = 0
                                    #loss_test_u_size1 = 0
                                    for i in range(len(sampler.eq_param)): #len(a)):
                                        d_ = model(var_test_data[i])
                                        d = torch.autograd.grad(d_,var_test_data[i] , grad_outputs=torch.ones_like(d_), create_graph=True )
                                        dx_total = 0
                                        for di in range(x_dim):
                                            dx1 = d[0][:,di].reshape(-1,1)
                                            dxx1 = torch.autograd.grad(dx1, var_test_data[i] , grad_outputs=torch.ones_like(dx1), create_graph=True )[0][:,di].reshape(-1,1)
                                            dx_total =  dx_total  +  dxx1[rnd_s_list[i]] 
                                            
                                        loss_test = loss_test + reg_p1*loss((dx_total).squeeze(), y[i].squeeze())
                                        #loss_test_u1 = loss_test_u1 +  torch.sum((d_[rnd_s_list[i]] - y_u[0])**2)
                                        #loss_test_u_size1 = loss_test_u_size1 + y_u[0].size(0)
                                        #print(loss_test_u_size1)
                                        #print(loss_test_u1)
                                    loss_test = loss_test/len(sampler.eq_param)

                                    
                                    
                                    
                                    if args.opt_method == 'Adam':
                                        if args.full == 0:
                                            x_b,y_b,rnd_sample_b  = sampler.rnd_sample_b(data_trpe = 'test')
                                        else:
                                            x_b,y_b,rnd_sample_b  = sampler.rnd_full_indexes_b(data_trpe = 'test')
                                    elif args.opt_method == 'LBFGS':
                                        x_b,y_b,rnd_sample_b  = sampler.rnd_full_indexes_b(data_trpe = 'test')
        
                                    for j in range(num_boundaries):
                                        temp = 0
                                        for i in range(len(sampler.eq_param)): #len(a)):
                                            d = model(x_b[(i-1)*num_boundaries+j])
                                            temp = temp + reg_p2*loss(d.squeeze(), y_b[(i-1)*num_boundaries+j].squeeze()) #+ reg_p3*loss(vx, vx_rhs) 
                                            
                                            #loss_test_u1 = loss_test_u1 + torch.sum((d - y_b[(k-1)*num_boundaries+j])**2)
                                            #loss_test_u_size1 = loss_test_u_size1 + y_b[(k-1)*num_boundaries+j].size(0)
                                            #print(loss_test_u_size1)
                                            #print(loss_test_u1)
                                    #loss_test_u1 = (loss_test_u1/loss_test_u_size1)**(0.5)
                                    #print(loss_test_u1)
                                            
                                            
                                    loss_test = loss_test + temp/len(sampler.eq_param)
                                    
                                    
                                    return loss_test.item(), loss_test_ #,acc_train.item()
               
                                def val_step(model, sampler ):
                                    model.eval()
                                    #with torch.no_grad():
                                    loss = nn.MSELoss()
                                    #model.eval()
                                    
                                    if args.opt_method == 'Adam':
                                        if args.full == 0:
                                            x,y,rnd_s_list  = sampler.rnd_sample(data_trpe = 'val')
                                        else:
                                            x,y,rnd_s_list  = sampler.full_indexes(data_trpe = 'val')
                                    elif args.opt_method == 'LBFGS':
                                        x,y,rnd_s_list = sampler.full_indexes(data_trpe = 'val')
                                        
                                    loss_val = 0
                                    for i in range(len(sampler.eq_param)): #len(a)):
                                        #print(rnd_s_list[i])
                                        #print(var_train_data[i])
                                        d = model(var_val_data[i])
                                        
                                        d = torch.autograd.grad(d,var_val_data[i] , grad_outputs=torch.ones_like(d), create_graph=True )
                                        dx_total = 0
                                        for di in range(x_dim):
                                            dx1 = d[0][:,di].reshape(-1,1)
                                            dxx1 = torch.autograd.grad(dx1, var_val_data[i] , grad_outputs=torch.ones_like(dx1), create_graph=True )[0][:,di].reshape(-1,1)
                                            dx_total =  dx_total  +  dxx1[rnd_s_list[i]] 
                                            
                                        loss_val = loss_val + reg_p1*loss((dx_total).squeeze(), y[i].squeeze())
                                    loss_val = loss_val/len(sampler.eq_param)
                                    
                                    
                                                                           
                                    if args.opt_method == 'Adam':
                                        if args.full == 0:
                                            x_b,y_b,rnd_sample_b  = sampler.rnd_sample_b(data_trpe = 'val')
                                        else:
                                            x_b,y_b,rnd_sample_b  = sampler.rnd_full_indexes_b(data_trpe = 'val')
                                    elif args.opt_method == 'LBFGS':
                                        x_b,y_b,rnd_sample_b  = sampler.rnd_full_indexes_b(data_trpe = 'val')
                                    
                                    for j in range(num_boundaries):
                                        temp = 0
                                        for i in range(len(sampler.eq_param)): #len(a)):
                                            d = model(x_b[(i-1)*num_boundaries+j])
                                            temp = temp + reg_p2*loss(d.squeeze(), y_b[(i-1)*num_boundaries+j].squeeze()) #+ reg_p3*loss(vx, vx_rhs) 
                                        #print(temp)
                                        loss_val = loss_val + temp/len(sampler.eq_param)
                                    
                                    #print("Val loss :" + str(loss_val.item()) )
            
                                    return loss_val.item() #,acc_train.item()
                              
                    
                                def train(): #datastr,splitstr):
                                    
                                    nn_input_list = nn.ModuleList()
                                    
                                    mlp_x_act = activation_ext(act = args.act_type_x).to(device)
                                    mlp_x= nn_mlp_last_na_custom_act(net_config_X1_, mlp_x_act).to(device)
                                    nn_input_list.append(mlp_x)
                                    len_input_concat = len(input_concat)
                                    

                                    
                                    for d in range(len_input_concat-2):
                                        mlp_quad_act2 = activation_ext(act = args.act_type_quad).to(device)
                                        mlp_quad2= nn_mlp_last_na_custom_act_tensor(net_config_X2_, mlp_quad_act2, mlp_quad_last_shape[d+1] ).to(device)
                                        nn_input_list.append(mlp_quad2)
                                    
                                    mlp_quad_act3 = activation_ext(act = args.act_type_quad).to(device) 
                                    mlp_quad3= nn_mlp_last_na_custom_act_tensor(net_config_X3_, mlp_quad_act3, mlp_quad_last_shape[len_input_concat-1]).to(device)
                                    nn_input_list.append(mlp_quad3)
                                    #mlp_quad= nn_mlp_custom_act(net_config_outer,mlp_quad_act)
                                    
                                    
                                    model = pinns_product_tensor(nn_input_list, mlp_quad_last_shape,input_concat).to(device)
                                    #model =Green_LR_mlp_gen_direct_low_rank(mlp_x,  r_func ,mlp_quad)
    
                                    if args.opt_method == 'Adam':
                                        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd_)
                                    elif args.opt_method == 'LBFGS':
                                        optimizer = optim.LBFGS(model.parameters(), history_size=args.history_size, max_iter=args.max_iter) #lr=lr, history_size=args.history_size, max_iter=args.max_iter, line_search_fn=args.line_search_fn)lr=lr, history_size=args.history_size, max_iter=args.max_iter, line_search_fn=args.line_search_fn)
                    
                                    bad_counter = 0
                                    best = 999999999
                                    for epoch in range(args.epochs):
                                        
                                        loss_tra = train_step(model,sampler, optimizer) # , vx_rhs)
                                        loss_val = val_step(model,sampler) #,  vx_rhs)
                                        #print("Train Loss: " + str(loss_tra))
                                        if np.mod(epoch,100) == 0:
                                            print("Train Loss: " + str(loss_tra) + " Validation Loss: " + str(loss_val))
                                        
                                        #'''
                                        if loss_val < best:
                                            best = loss_val
                                            torch.save(model.state_dict(), checkpt_file)
                                            bad_counter = 0
                                        else:
                                            bad_counter += 1
                    
                                        if bad_counter == args.patience:
                                            break
                                        #'''
                                    
                                    model.load_state_dict(torch.load(checkpt_file))
                                    loss_val = val_step(model,sampler) #, vx_rhs)
                                    ##loss_train = tr_step(model,a, train_data,train_label, train_data_b ,train_label_b)
                                    loss_test, loss_test_  = test_step(model,sampler)
                                    
                                    
                                    return loss_val,loss_test , loss_test_ #acc_val
                                
            
    
                                loss_val,loss_test,loss_test_= train() 
                                res_array[wd.index(wd_),  drop.index(drop_),  reg_para1.index(reg_p1), reg_para2.index(reg_p2),      j] = loss_val
                                
                                filename = './Poisson3D_gridsearch_homogen_new_pinns/TT_sep/'+str(args.data) + '_' + str(args.seed)   + '_' + str(lr)     + '_' + str(wd)  + '_' + str(drop)   + '_' + str(reg_p1)   + '_' + str(reg_p2)   + '_' + str(net_config_X1)   + '_'   + str(net_config_X2)  + '_'   + str(net_config_X3)    + '_'   + str(r)    + '_' + str(args.act_type_x) + '_' + str(args.act_type_quad)   + '_' + str(args.act_type_last)  + '_' + str(loss_val)  + '_' + str(loss_test)  + '.mat'
    
                                sio.savemat(filename, {'res': res_array})
                                print(res_array)
                                
