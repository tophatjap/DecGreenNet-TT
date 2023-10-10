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
from matplotlib import cm


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
parser.add_argument('--r', type=int, default=2, help='number of quadrature points')
parser.add_argument('--r1', type=int, default=3, help='number of quadrature points')
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
reg_para2 = [1 ] 
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
#net_config_X_arr.append([x_dim, 256,256,256,256,  r1])
net_config_X_arr.append([x_dim, 512,512,512,512,  r1])


net_config_outer_arr1 = list()
net_config_outer_arr1.append([qx_dim, 64,64,64,r1*r])
#net_config_outer_arr1.append([qx_dim, 128,128,r1*r])


net_config_outer_arr_last = list()
net_config_outer_arr_last.append([qx_dim, 64,64,r])
#net_config_outer_arr_last.append([qx_dim, 64,64,64,r])



fname = 'data_new/poisson2D_multi_[10-100].mat'

  

res_array = np.zeros([len(wd) ,len(drop) ,len(reg_para1)  ,len(reg_para2)  ,  no_samples])
cudaid =  "cpu" # "cuda:" + str(args.dev)  #
device = torch.device(cudaid)
checkpt_file = 'pretrained/' + uuid.uuid4().hex + '.pt'
print(cudaid, checkpt_file)

#samples
sampler = Data_Sampler(fname,batch_size=batch_size, num_b=num_boundaries, device=device)
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

# Generate Monte-Carlo Samples 
q_x = list()
quad_x = list() 
green_x = list() 
for p in range(x_dim):
    q_x1 = np.random.rand(num_quad,1)
    quad_x1 = torch.FloatTensor(q_x1).to(device)
    q_x.append(q_x1)
    quad_x.append(quad_x1)
    q_0 = torch.FloatTensor([0.5]).to(device)
    green_x.append(q_0)

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
                                            x,y,rnd_s_list  = sampler.rnd_sample(data_trpe = 'train')
                                        else:
                                            x,y,rnd_s_list  = sampler.full_indexes(data_trpe = 'train')
                                    elif args.opt_method == 'LBFGS':
                                        x,y,rnd_s_list = sampler.full_indexes(data_trpe = 'train')
                                    
                                    loss_train = 0
                                    for i in range(len(sampler.eq_param)): #len(a)):
                                        #print(rnd_s_list[i])
                                        #print(var_train_data[i])
                                        d,d_green = model(var_train_data[i],sampler.eq_param[i],quad_x,green_x)
                                        
                                        d = torch.autograd.grad(d,var_train_data[i] , grad_outputs=torch.ones_like(d), create_graph=True )
                                        dx_total = 0
                                        for di in range(x_dim):
                                            dx1 = d[0][:,di].reshape(-1,1)
                                            dxx1 = torch.autograd.grad(dx1, var_train_data[i] , grad_outputs=torch.ones_like(dx1), create_graph=True )[0][:,di].reshape(-1,1)
                                            dx_total =  dx_total  +  dxx1[rnd_s_list[i]] 
                                            
                                        loss_train = loss_train + reg_p1*loss(-(dx_total).squeeze(), y[i].squeeze())
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
                                            d,d_green = model(x_b[(i-1)*num_boundaries+j], sampler.eq_param[i], quad_x,green_x)
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
                                        loss_test_u = 0
                                        loss_test_u_size = 0
                                        #for i in range(len(sampler.eq_param)): #len(a)):
                                        d_,d_green = model(var_test_data[k],sampler.eq_param[k],quad_x,green_x)
                                        d = torch.autograd.grad(d_,var_test_data[k] , grad_outputs=torch.ones_like(d_), create_graph=True )
                                        
                                        dx_total = 0
                                        for di in range(x_dim):
                                            dx1 = d[0][:,di].reshape(-1,1)
                                            dxx1 = torch.autograd.grad(dx1, var_test_data[k] , grad_outputs=torch.ones_like(dx1), create_graph=True )[0][:,di].reshape(-1,1)
                                            dx_total =  dx_total  +  dxx1[rnd_s_list[k]] 
                                            
                                        loss_test = loss_test + reg_p1*loss(-(dx_total).squeeze(), y[k].squeeze())
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
                                            d,d_green = model(x_b[(k-1)*num_boundaries+j], sampler.eq_param[k], quad_x,green_x)
                                            temp = temp + reg_p2*loss(d.squeeze(), y_b[(k-1)*num_boundaries+j].squeeze()) #+ reg_p3*loss(vx, vx_rhs) 
                                            loss_test = loss_test + temp
                                            loss_test_.append(loss_test.item())
                                            
                                            #loss_test_u = loss_test_u + torch.sum((d - y_b[(k-1)*num_boundaries+j])**2)
                                            #loss_test_u_size = loss_test_u_size + y_b[(k-1)*num_boundaries+j].size(0)
                                            #print(loss_test_u_size)
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
                                    loss_test_u1 = 0
                                    loss_test_u_size1 = 0
                                    for i in range(len(sampler.eq_param)): #len(a)):
                                        d_,d_green = model(var_test_data[i],sampler.eq_param[i],quad_x,green_x)
                                        d = torch.autograd.grad(d_,var_test_data[i] , grad_outputs=torch.ones_like(d_), create_graph=True )
                                        dx_total = 0
                                        for di in range(x_dim):
                                            dx1 = d[0][:,di].reshape(-1,1)
                                            dxx1 = torch.autograd.grad(dx1, var_test_data[i] , grad_outputs=torch.ones_like(dx1), create_graph=True )[0][:,di].reshape(-1,1)
                                            dx_total =  dx_total  +  dxx1[rnd_s_list[i]] 
                                            
                                        loss_test = loss_test + reg_p1*loss(-(dx_total).squeeze(), y[i].squeeze())
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
                                            d,d_green = model(x_b[(i-1)*num_boundaries+j], sampler.eq_param[i], quad_x,green_x)
                                            temp = temp + reg_p2*loss(d.squeeze(), y_b[(i-1)*num_boundaries+j].squeeze()) #+ reg_p3*loss(vx, vx_rhs) 
                                            
                                            #loss_test_u1 = loss_test_u1 + torch.sum((d - y_b[(k-1)*num_boundaries+j])**2)
                                            #loss_test_u_size1 = loss_test_u_size1 + y_b[(k-1)*num_boundaries+j].size(0)
                                            #print(loss_test_u_size1)
                                            #print(loss_test_u1)
                                    #loss_test_u1 = (loss_test_u1/loss_test_u_size1)**(0.5)
                                    #print(loss_test_u1)
                                            
                                            
                                    loss_test = loss_test + temp/len(sampler.eq_param)
                                    
                                     
                                    return loss_test.item(), loss_test_ #, loss_test_u.item(), loss_test_u1.item() #,acc_train.item()
               
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
                                        d ,d_green = model(var_val_data[i],sampler.eq_param[i],quad_x,green_x)
                                        
                                        d = torch.autograd.grad(d,var_val_data[i] , grad_outputs=torch.ones_like(d), create_graph=True )
                                        dx_total = 0
                                        for di in range(x_dim):
                                            dx1 = d[0][:,di].reshape(-1,1)
                                            dxx1 = torch.autograd.grad(dx1, var_val_data[i] , grad_outputs=torch.ones_like(dx1), create_graph=True )[0][:,di].reshape(-1,1)
                                            dx_total =  dx_total  +  dxx1[rnd_s_list[i]] 
                                            
                                        loss_val = loss_val + reg_p1*loss(-(dx_total).squeeze(), y[i].squeeze())
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
                                            d,d_green = model(x_b[(i-1)*num_boundaries+j], sampler.eq_param[i], quad_x,green_x)
                                            temp = temp + reg_p2*loss(d.squeeze(), y_b[(i-1)*num_boundaries+j].squeeze()) #+ reg_p3*loss(vx, vx_rhs) 
                                        #print(temp)
                                        loss_val = loss_val + temp/len(sampler.eq_param)
                                    
                                    #print("Val loss :" + str(loss_val.item()) )
            
                                    return loss_val.item() #,acc_train.item()
                              
                    
                                def train(): #datastr,splitstr):
                                    
                                    mlp_x_act = activation_ext(act = args.act_type_x).to(device)
                                    mlp_x= nn_mlp_last_na_custom_act(net_config_X, mlp_x_act).to(device)
                                    
                                    nn_quad_list = nn.ModuleList()
                                    len_quad_concat = len(quad_concat)
                                    
                                    mlp_quad_act1 = activation_ext(act = args.act_type_quad).to(device)
                                    mlp_quad1= nn_mlp_last_na_custom_act_tensor(net_config_outer1, mlp_quad_act1, mlp_quad_last_shape[0] ).to(device)
                                    nn_quad_list.append(mlp_quad1)
                                    
                                    
                                    mlp_quad_act3 = activation_ext(act = args.act_type_quad).to(device) 
                                    mlp_quad3= nn_mlp_last_na_custom_act_tensor(net_config_outer_last, mlp_quad_act3, mlp_quad_last_shape[len_quad_concat-1]).to(device)
                                    nn_quad_list.append(mlp_quad3)

                                    r_func =PoiXd_a_tensor_sep(x_dim).to(device)
                                                                        
                                    model = DecGreenNet_product_tensor_sep_poisson_modnet(mlp_x,  r_func ,nn_quad_list, mlp_quad_last_shape,quad_concat).to(device)
    
                                    if args.opt_method == 'Adam':
                                        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd_)
                                    elif args.opt_method == 'LBFGS':
                                        optimizer = optim.LBFGS(model.parameters(), history_size=args.history_size, max_iter=args.max_iter) 
                    
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
                                    
                                    
                                    ## plot --------------------------------------------------------------------------
                                    a_ = 15
                                    x = np.linspace(0,1,101)
                                    y = np.linspace(0,1,101)
    
                                    quad_x1 = torch.FloatTensor(torch.from_numpy(x).float()).to(device)
                                    quad_y1 = torch.FloatTensor(torch.from_numpy(y).float()).to(device)
                                    x_ = np.zeros([1,2])
                                    u = np.zeros([101,101])
                                    u_green = np.zeros([101,101])
                                    u1 = np.zeros([101,101])
                                    for i in range(101):
                                        for j in range(101):
                                            x_[0,0] = x[i]
                                            x_[0,1] = y[j]
                                            u[i,j] =  0.5*a_*x[i]*(x[i]-1)*x[j]*(x[j]-1)
                                            xx_ = torch.FloatTensor(torch.from_numpy(x_).float()).to(device)
                                            dump_ , u_green[i,j] = model(xx_,1,quad_x,green_x)
                                            u1[i,j], dump = model(xx_,a_,quad_x,green_x)
                                    
                                    plot_str = 'Poisson2D_gridsearch_mult_sep_new/plot/' + '_' + str(net_config_X)   + '_'   + str(net_config_outer1) + '_'   + str(net_config_outer_last)   + '_'   + str(r)  + '_'   + str(r1)
                                    #ext_name = '_' + str(r) + '_' + str(num_quad)
                                    
                                    X, Y = np.meshgrid(x, y)
    
                                    fig = plt.figure()
                                    ax = fig.add_subplot(1, 1, 1, projection='3d')
                                 
                                    surf = ax.plot_surface(X, Y, u_green, rstride=1, cstride=1, cmap=cm.coolwarm,
                                                           linewidth=0, antialiased=False)
                                    ax.set_zlim(0, 0.40)
                                    
                                    fig.colorbar(surf, shrink=0.5, aspect=10)
                                    fig.show()
                                    fig.savefig(plot_str + "green.eps",bbox_inches='tight')
                                    #print(u1)
                                    
                                    fig = plt.figure()
                                    plt.imshow(u1,cmap=cm.coolwarm)
                                    plt.colorbar()
                                    #plt.scatter(x, y, s=u, alpha=0.5)
                                    plt.xlabel('x')
                                    plt.ylabel('y')
                                    plt.xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90], [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]) #np.arange(0, 100, step=10))
                                    plt.yticks([90, 80, 70, 60, 50, 40, 30, 20, 10,0], [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])                             
                                    plt.show()
                                                           
                                    fig.savefig(plot_str + '_predicted' + '.eps' ,bbox_inches='tight')
                                    #print(u1)
                                    
                                    
                                    u_ = u - u1
                                    print(u_)
                                    fig = plt.figure()
                                    #ax = fig.add_subplot(1, 1, 1, projection='2d')
                                    plt.imshow(u_,cmap=cm.coolwarm)
                                    plt.colorbar() #cax=cax)
                                    #plt.scatter(x, y, s=u, alpha=0.5)
                                    plt.xlabel('x')
                                    plt.ylabel('y')
                                    plt.xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90], [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]) #np.arange(0, 100, step=10))
                                    plt.yticks([90, 80, 70, 60, 50, 40, 30, 20, 10,0], [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
                                    #plt.savefig(plot_str + 'poi2d_multi_dec_prod_diff_' + ext_name + '.eps',bbox_inches='tight')
                                    plt.show()
                         
                                    fig.savefig(plot_str + '_error' + '.eps' ,bbox_inches='tight')
                                    
                                    fig = plt.figure()
                                    #ax = fig.add_subplot(1, 1, 1, projection='2d')
                                    plt.imshow(u,cmap=cm.coolwarm)
                                    plt.colorbar() #cax=cax)
                                    #plt.scatter(x, y, s=u, alpha=0.5)
                                    plt.xlabel('x')
                                    plt.ylabel('y')
                                    plt.xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90], [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]) #np.arange(0, 100, step=10))
                                    plt.yticks([90, 80, 70, 60, 50, 40, 30, 20, 10,0], [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
                                    #plt.savefig(plot_str + 'poi2d_' + ext_name + '.eps',bbox_inches='tight')
                                    plt.show()
                                    fig.savefig(plot_str + '_exact' + '.eps' ,bbox_inches='tight')
                                    #W'''
                                    
                                    
                                    return loss_val,loss_test , loss_test_ #acc_val
                                
            
                                
                                loss_val,loss_test,loss_test_= train() 
                                str_summary = "validation error: " + str(loss_val) + " Prediction error: " + + str(loss_val)
                                 print(str_summary)
                                res_array[wd.index(wd_),  drop.index(drop_),  reg_para1.index(reg_p1), reg_para2.index(reg_p2),      j] = loss_val
                                
                                # Specift a location to store resutls or comenet 
                                filename = './Poisson2D_gridsearch_mult_sep_new/tt/'+str(args.data) + '_' + str(args.seed)   + '_' + str(lr)     + '_' + str(wd)  + '_' + str(drop)   + '_' + str(reg_p1)   + '_' + str(reg_p2)   + '_' + str(net_config_X)   + '_'   + str(net_config_outer1) + '_'   + str(net_config_outer_last)   + '_'   + str(r)  + '_'   + str(r1)  + '_'   + str(num_quad)  + '_' + str(args.act_type_x) + '_' + str(args.act_type_quad)   + '_' + str(args.act_type_last)  + '_' + str(loss_val)  + '_' + str(loss_test) + '.mat'
    
                                sio.savemat(filename, {'res': res_array})
                                print(res_array)
                                
