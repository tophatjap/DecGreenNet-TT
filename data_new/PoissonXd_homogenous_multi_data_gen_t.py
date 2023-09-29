#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 13:24:57 2022

@author: kishan
"""

import math
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from auxilary_functions import *
import itertools

train_size =  10000 #100
test_size = 3000 #50
val_size = 3000 #50

train_size_b = 2000
test_size_b = 1000
val_size_b = 1000

x_range = [0,1]
y_range = [0,1]

x_grid = 101
y_grid = 101

a = [15]  #[10,20,30,40,50,60,70,80,90,100] # [10,20,30,40,50] # [10,12,14,16,18,20]  #[10,20,30,40,50,60,70,80,90,100]

num_dim = 2
#num_perm = 6 #num_dim*2 #+ num_dim*(num_dim-1)
#print(num_perm)
# Generate data
# Train

train_data = list()
test_data = list()
val_data = list()
train_label = list()
test_label = list()
test_label_u = list()
val_label = list()
val_label_u = list()

data_random_main = list()
for i in range(len(a)):
    data_random_main.append(np.random.random_sample([ train_size + test_size + val_size,num_dim]) )
    train_data.append(np.zeros([ train_size,num_dim]))
    test_data.append(np.zeros([ test_size,num_dim]))
    val_data.append(np.zeros([ val_size,num_dim]))
    train_label.append(np.zeros([ train_size]))
    test_label.append(np.zeros([ test_size]))
    test_label_u.append(np.zeros([ test_size]))
    val_label.append(np.zeros([ val_size]))
    val_label_u.append(np.zeros([ val_size]))

#print(data_random_main[0][0][0])
#print(data_random_main[0][1])

for l in range(len(a)):
    k =0
    for i in range(train_size):
        #print(k)
        for d in range(num_dim):
            train_data[l][i][d] = (data_random_main[l][k][d] - 0.5)*2
        k = k + 1
    train_label[l] =  Poisson_homogenous_dx(train_data[l])

    for i in range(test_size):
        #print(k)
        for d in range(num_dim):
            test_data[l][i][d] = (data_random_main[l][k][d] - 0.5)*2
        k = k + 1
    test_label[l] =  Poisson_homogenous_dx(test_data[l])
    test_label_u[l] = Poisson_homogenous(test_data[l])

    for i in range(val_size):
        for d in range(num_dim):
            val_data[l][i][d] = (data_random_main[l][k][d] - 0.5)*2
        k = k + 1
    val_label[l] = Poisson_homogenous_dx(val_data[l]) 
    val_label_u[l] = Poisson_homogenous(val_data[l]) 


## Boundary data
perms =  ["".join(seq) for seq in itertools.product("01", repeat=num_dim-1)]
num_perm = num_dim*len(perms)
len_perm = len(perms)
print(len(perms))
print(num_perm)
print(len_perm)
print(perms)

bound_rand_data = list()
for i in range(num_perm):
    bound_rand_data.append(np.random.random_sample([len(a), (train_size_b + test_size_b + val_size_b) ,1]))
    
    
train_data_b = list()
test_data_b = list()
val_data_b = list()
train_label_b = list()
test_label_b = list()
val_label_b = list()
for i in range(num_perm):
    train_data_b.append(np.zeros([len(a), train_size_b,num_dim]))
    test_data_b.append(np.zeros([len(a), test_size_b,num_dim]))
    val_data_b.append(np.zeros([len(a), val_size_b,num_dim]))
    train_label_b.append(np.zeros([len(a), train_size_b]))
    test_label_b.append(np.zeros([len(a), test_size_b]))
    val_label_b.append(np.zeros([len(a), val_size_b]))

# * 0 0
for l in range(len(a)):
    for j in range(num_dim):
        for p in range(len_perm):
            k = 0
            for v in range(train_size_b):
                cc = 0
                for d in range(num_dim):
                    if j == d: 
                        train_data_b[j*(len_perm)+p][l][v][d] = (bound_rand_data[j*(len_perm)+p][l][k][0] - 0.5)*2
                    else:
                        print(int(perms[p][cc]))
                        train_data_b[j*(len_perm)+p][l][v][d] = (int(perms[p][cc]) - 0.5)*2
                        cc = cc +1
                k=k+1

            train_label_b[j*(len_perm)+p][l] = Poisson_homogenous(train_data_b[j*(len_perm)+p][l])
                
            for v in range(test_size_b):
                cc = 0
                for d in range(num_dim):
                    if j == d: 
                        test_data_b[j*(len_perm)+p][l][v][d] = (bound_rand_data[j*(len_perm)+p][l][k][0] - 0.5)*2
                    else:
                        print(int(perms[p][cc]))
                        test_data_b[j*(len_perm)+p][l][v][d] = (int(perms[p][cc]) - 0.5)*2
                        cc = cc +1
                k=k+1
            test_label_b[j*(len_perm)+p][l] = Poisson_homogenous(test_data_b[j*(len_perm)+p][l])
                
            for v in range(val_size_b):
                cc = 0
                for d in range(num_dim):
                    if j == d: 
                        val_data_b[j*(len_perm)+p][l][v][d] = (bound_rand_data[j*(len_perm)+p][l][k][0] - 0.5)*2
                    else:
                        print(int(perms[p][cc]))
                        val_data_b[j*(len_perm)+p][l][v][d] = (int(perms[p][cc]) - 0.5)*2
                        cc = cc +1
                k=k+1
            val_label_b[j*(len_perm)+p][l] = Poisson_homogenous(val_data_b[j*(len_perm)+p][l])
                
print(train_data_b) 
print(train_label_b) 
print(test_data_b)
print(test_label_b)  
print(val_data_b) 
print(val_label_b) 


print(val_label)
print(test_label_u)
    
data = {'a': a, 'train_data': train_data, 'train_label':  train_label,'test_data': test_data, 'test_label':  test_label,'val_data': val_data, 'val_label':  val_label, 'train_data_b': train_data_b, 'train_label_b':  train_label_b,'test_data_b': test_data_b, 'test_label_b':  test_label_b,'val_data_b': val_data_b,'val_label_b': val_label_b, 'test_label_u':test_label_u , 'val_label_u': val_label_u}
#data = {'a': a, 'train_data': train_data, 'train_label':  train_label,'test_data': test_data, 'test_label':  test_label,'val_data': val_data, 'val_label':  val_label}

filename = './poisson_homogenous' + str(num_dim) + '_num_boundaries_' + str(num_perm) + '_D_no_boundary_'  + '.mat' # './poisson4D_multi_[10-100].mat'
# filename = 'sims.mat'
sio.savemat(filename, {'data': data})

#train_data =
#train_label = 
