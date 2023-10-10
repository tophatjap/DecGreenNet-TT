#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import random
import argparse
import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from model_higher_order_sep import * 
from equation_higher_order_models_sep import * 
from activations_ext import * 
from data_sampler import * 
import matplotlib.pyplot as plt
import scipy.io as sio
import uuid
import torch.optim as optim
from torch.autograd import Variable
import time

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
parser.add_argument('--num_quad', type=int, default=20, help='number of quadrature points')
parser.add_argument('--r', type=int, default=3, help='number of quadrature points')
parser.add_argument('--r1', type=int, default=6, help='number of quadrature points')
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
reg_para2 = [1,10,100] #,0.01,0.0001] # 1e-2,1e-3,1e-4]
epoch = args.epochs
x_dim = 2
qx_dim = 1
r = args.r
r1 = args.r1
num_quad=args.num_quad
num_boundaries = 4

mlp_quad_last_shape = [[r1,r],[r,1]]
quad_concat = [[0],[1]]
tensor_quad_dim = [num_quad,num_quad,num_quad]
no_samples = 1




net_config_X_arr = list()
net_config_X_arr.append([x_dim, 512,512,512,  r1])
#net_config_X_arr.append([x_dim, 512,512,  r1])


net_config_outer_arr1 = list()

net_config_outer_arr1.append([qx_dim, 32,32,32,r1*r])
net_config_outer_arr1.append([qx_dim, 64,64,r1*r])
net_config_outer_arr1.append([qx_dim, 64,64,64,r1*r])
#net_config_outer_arr1.append([qx_dim, 128,128,r1*r])


net_config_outer_arr_last = list()

net_config_outer_arr_last.append([qx_dim, 32,32,32,r])
net_config_outer_arr_last.append([qx_dim, 64,64,r])
net_config_outer_arr_last.append([qx_dim, 64,64,64,r])



fname = 'data_new/poisson_homogenous2_num_boundaries_4_D_boundary_.mat' #' poisson_homogenous3_num_boundaries_12_D_no_boundary_.mat'

  

res_array = np.zeros([len(wd) ,len(drop) ,len(reg_para1)  ,len(reg_para2)  ,  no_samples])
cudaid =  "cuda:" + str(args.dev)  # "cpu"
device = torch.device(cudaid)
checkpt_file = 'pretrained/' + uuid.uuid4().hex + '.pt'
print(cudaid, checkpt_file)

#samples
sampler = Data_Sampler2(fname,batch_size=batch_size, num_b=4, device=device)
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
    q_x1 = np.random.rand(num_quad,1)
    quad_x1 = torch.FloatTensor(q_x1).to(device)
    q_x.append(q_x1)
    quad_x.append(quad_x1)


for net_config_X in net_config_X_arr:
    for net_config_outer1 in net_config_outer_arr1:
        for net_config_outer_last in net_config_outer_arr_last:
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
                                            x,y,y_u,rnd_s_list  = sampler.rnd_sample(data_trpe = 'train')
                                        else:
                                            x,y,y_u,rnd_s_list  = sampler.full_indexes(data_trpe = 'train')
                                    elif args.opt_method == 'LBFGS':
                                        x,y,y_u,rnd_s_list = sampler.full_indexes(data_trpe = 'train')
                                    
                                    loss_train = 0
                                    for i in range(len(sampler.eq_param)): #len(a)):
                                        #print(rnd_s_list[i])
                                        #print(var_train_data[i])
                                        d = model(var_train_data[i],sampler.eq_param[i],quad_x)
                                        
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
                                            
                                            d = model(x_b[(i-1)*num_boundaries+j], sampler.eq_param[i], quad_x)
                                            
                                            temp = temp + reg_p2*loss(d.squeeze(), y_b[(i-1)*4+j].squeeze()) #+ reg_p3*loss(vx, vx_rhs) 
                                        #print(temp)
                                        loss_train = loss_train + temp/len(sampler.eq_param)
                                   
                                    
                                    #print("Train loss :" + str(loss_train.item()) )
                                    loss_train.backward()
                                    optimizer.step()
                                    return loss_train.item() #,acc_train.item()
                    
                                def test_step(model,sampler):
                                    model.eval()
                                    
                                    eq  = PoiXd_homogen_sol()
                                    
                                    y_pred  =  [] #model(var_test_data[k],sampler.eq_param[k],quad_x)
                                    y_exact = [] 
                                    error   =   [] # y_pred - y_exact
                                    
                                    x_b,y_b,rnd_sample_b  = sampler.rnd_full_indexes_b(data_trpe = 'test')
                                    for k in range(len(sampler.eq_param)):
                                        
                                        y_pred = np.concatenate((model(var_test_data[k],sampler.eq_param[k],quad_x).cpu().detach().numpy(), y_pred))
                                        y_exact = np.concatenate( (eq(var_test_data[k],sampler.eq_param[k]).cpu().detach().numpy() , y_exact))
                                        
                                        for j in range(num_boundaries):
                                            y_pred = np.concatenate( (model(x_b[(k-1)*num_boundaries+j], sampler.eq_param[k], quad_x).cpu().detach().numpy() , y_pred) )
                                            y_exact =   np.concatenate( (eq( x_b[(k-1)*num_boundaries+j], sampler.eq_param[k] ).cpu().detach().numpy() , y_exact)) 
                                            #print(len(y_pred))
                                   
                                    error_abs = np.sqrt( (np.sum((y_pred - y_exact)**2))/len(y_pred))
                                    
                                    
                                    return error_abs 
               
                                def val_step(model, sampler ):
                                    model.eval()
                                    #with torch.no_grad():
                                    loss = nn.MSELoss()
                                    #model.eval()
                                    
                                    if args.opt_method == 'Adam':
                                        if args.full == 0:
                                            x,y,y_u,rnd_s_list  = sampler.rnd_sample(data_trpe = 'val')
                                        else:
                                            x,y,y_u,rnd_s_list  = sampler.full_indexes(data_trpe = 'val')
                                    elif args.opt_method == 'LBFGS':
                                        x,y,y_u,rnd_s_list = sampler.full_indexes(data_trpe = 'val')
                                        
                                    loss_val = 0
                                    for i in range(len(sampler.eq_param)): #len(a)):
                                        
                                        d = model(var_val_data[i],sampler.eq_param[i],quad_x)
                                        
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
                                            d = model(x_b[(i-1)*num_boundaries+j], sampler.eq_param[i], quad_x)
                                            temp = temp + reg_p2*loss(d.squeeze(), y_b[(i-1)*num_boundaries+j].squeeze()) #+ reg_p3*loss(vx, vx_rhs) 
                                        #print(temp)
                                        loss_val = loss_val + temp/len(sampler.eq_param)
                                    
                                    
            
                                    return loss_val.item() 
                              
                    
                                def train():
                                    
                                    mlp_x_act = activation_ext(act = args.act_type_x).to(device)
                                    mlp_x= nn_mlp_last_na_custom_act(net_config_X, mlp_x_act).to(device)
                                    
                                    len_quad_concat = len(quad_concat)
                                    nn_quad_list = nn.ModuleList()
                                    mlp_quad_act1 = activation_ext(act = args.act_type_quad).to(device)
                                    mlp_quad1= nn_mlp_last_na_custom_act_tensor(net_config_outer1, mlp_quad_act1, mlp_quad_last_shape[0] ).to(device)
                                    nn_quad_list.append(mlp_quad1)
                                    

                                    
                                    mlp_quad_act3 = activation_ext(act = args.act_type_quad).to(device) 
                                    mlp_quad3= nn_mlp_last_na_custom_act_tensor(net_config_outer_last, mlp_quad_act3, mlp_quad_last_shape[len_quad_concat-1]).to(device)
                                    nn_quad_list.append(mlp_quad3)
                                    
                                    r_func =PoiXd_homogen_tensor_sep(quad_concat).to(device)
                                    
                                    model = DecGreenNet_product_tensor_sep(mlp_x,  r_func ,nn_quad_list, mlp_quad_last_shape,quad_concat).to(device)
                                    
    
                                    if args.opt_method == 'Adam':
                                        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd_)
                                    elif args.opt_method == 'LBFGS':
                                        optimizer = optim.LBFGS(model.parameters(), history_size=args.history_size, max_iter=args.max_iter) #lr=lr, history_size=args.history_size, max_iter=args.max_iter, line_search_fn=args.line_search_fn)lr=lr, history_size=args.history_size, max_iter=args.max_iter, line_search_fn=args.line_search_fn)
                    
                                    start = time.time()
                                    bad_counter = 0
                                    best = 999999999
                                    for epoch in range(args.epochs):
                                        
                                        loss_tra = train_step(model,sampler, optimizer) # , vx_rhs)
                                        loss_val = val_step(model,sampler) #,  vx_rhs)
                                        #print("Train Loss: " + str(loss_tra))
                                        if np.mod(epoch,100) == 0:
                                            print("Train Loss: " + str(loss_tra) + " Validation Loss: " + str(loss_val))
                                        
                                        
                                        if loss_val < best:
                                            best = loss_val
                                            torch.save(model.state_dict(), checkpt_file)
                                            bad_counter = 0
                                        else:
                                            bad_counter += 1
                    
                                        if bad_counter == args.patience:
                                            break
                                        
                                    
                                    t = time.time() - start
                                    
                                    model.load_state_dict(torch.load(checkpt_file))
                                    loss_val = val_step(model,sampler) #, vx_rhs)
                                    ##loss_train = tr_step(model,a, train_data,train_label, train_data_b ,train_label_b)
                                    error_abs = test_step(model,sampler)
                                    
                                    
                                    return error_abs, t 
                                
            
    
                                error_abs,  t = train() 
                                str_summary = "validation error: " + str(loss_val) + " Prediction error: " + + str(error_abs) + " Training time: " + + str(t)
                                print(str_summary)
                                res_array[wd.index(wd_),  drop.index(drop_),  reg_para1.index(reg_p1), reg_para2.index(reg_p2),      j] = error_abs
                                
                                filename = './Poisson2D_gridsearch_homogen_final/TT/'+str(args.data) + '_' + str(args.seed)   + '_' + str(lr)     + '_' + str(wd)  + '_' + str(drop)   + '_' + str(reg_p1)   + '_' + str(reg_p2)   + '_' + str(net_config_X)   + '_'   + str(net_config_outer1)     + '_'   + str(net_config_outer_last)   + '_'   + str(r)  + '_'   + str(r1)  + '_'   + str(num_quad)  + '_' + str(args.act_type_x) + '_' + str(args.act_type_quad)   + '_' + str(args.act_type_last)  + '_' + str(error_abs)  + '_' + str(error_abs)  + '_time_' + str(t)  + '.mat'
    
                                sio.savemat(filename, {'res': res_array})
                                print(res_array)
                                
