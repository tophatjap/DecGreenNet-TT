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

train_size = 10000 #100
test_size = 5000 #50
val_size =5000 #50

train_size_b = 2000
test_size_b = 1000
val_size_b = 1000

x_range = [0,1]
y_range = [0,1]

x_grid = 101
y_grid = 101

a = [15]  #[10,20,30,40,50,60,70,80,90,100] # [10,20,30,40,50] # [10,12,14,16,18,20]  #[10,20,30,40,50,60,70,80,90,100]

num_dim = 12
num_perm = num_dim*2 #+ num_dim*(num_dim-1)
print(num_perm)
# Generate data
# Train

train_data = list()
test_data = list()
val_data = list()
train_label = list()
test_label = list()
val_label = list()

data_random_main = list()
for i in range(len(a)):
    data_random_main.append(np.random.random_sample([ train_size + test_size + val_size,num_dim]) )
    train_data.append(np.zeros([ train_size,num_dim]))
    test_data.append(np.zeros([ test_size,num_dim]))
    val_data.append(np.zeros([ val_size,num_dim]))
    train_label.append(np.zeros([ train_size]))
    test_label.append(np.zeros([ test_size]))
    val_label.append(np.zeros([ val_size]))

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

    for i in range(val_size):
        for d in range(num_dim):
            val_data[l][i][d] = (data_random_main[l][k][d] - 0.5)*2
        k = k + 1
    val_label[l] = Poisson_homogenous_dx(val_data[l]) 


## Boundary data
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


for l in range(len(a)):
    k = 0
    for i in range(train_size_b):
        for d in range(num_dim):
            train_data_b[d][l][i][d] = bound_rand_data[d][l][k][0]
        k = k + 1
    for d in range(num_dim):
        train_label_b[d][l] = Poisson_homogenous(train_data_b[d][l])
    
    
    for i in range(test_size_b):
        for d in range(num_dim):
            test_data_b[d][l][i][d] = bound_rand_data[d][l][k][0]
        k = k + 1
    for d in range(num_dim):
        test_label_b[d][l] = Poisson_homogenous(test_data_b[d][l])
    
    for i in range(val_size_b):
        for d in range(num_dim):
            val_data_b[d][l][i][d] = bound_rand_data[d][l][k][0]
        k = k + 1
    for d in range(num_dim):
        val_label_b[d][l] = Poisson_homogenous(val_data_b[d][l])    
    #print(val_data_b)

for l in range(len(a)):
    k = 0
    for i in range(train_size_b):
        for d in range(num_dim):
            train_data_b[num_dim+d][l][i] = np.ones(num_dim)
            train_data_b[num_dim+d][l][i][d] = bound_rand_data[num_dim+d][l][k][0]
        k = k + 1
    for d in range(num_dim):
        train_label_b[num_dim+d][l] = Poisson_homogenous(train_data_b[num_dim+d][l])

    for i in range(test_size_b):
        for d in range(num_dim):
            test_data_b[num_dim+d][l][i] = np.ones(num_dim)
            test_data_b[num_dim+d][l][i][d] = bound_rand_data[num_dim+d][l][k][0]
        k = k + 1
    for d in range(num_dim):
        test_label_b[num_dim+d][l] = Poisson_homogenous(test_data_b[num_dim+d][l])

    for i in range(val_size_b):
        for d in range(num_dim):
            val_data_b[num_dim+d][l][i] = np.ones(num_dim)
            val_data_b[num_dim+d][l][i][d] = bound_rand_data[num_dim+d][l][k][0]
        k = k + 1
    for d in range(num_dim):
        val_label_b[num_dim+d][l] = Poisson_homogenous(val_data_b[num_dim+d][l])

print(val_label)

    
data = {'a': a, 'train_data': train_data, 'train_label':  train_label,'test_data': test_data, 'test_label':  test_label,'val_data': val_data, 'val_label':  val_label, 'train_data_b': train_data_b, 'train_label_b':  train_label_b,'test_data_b': test_data_b, 'test_label_b':  test_label_b,'val_data_b': val_data_b,'val_label_b': val_label_b}
#data = {'a': a, 'train_data': train_data, 'train_label':  train_label,'test_data': test_data, 'test_label':  test_label,'val_data': val_data, 'val_label':  val_label}

filename = './poisson_homogenous' + str(num_dim) + 'D_multi_' + str(a)  + '.mat' # './poisson4D_multi_[10-100].mat'
# filename = 'sims.mat'
sio.savemat(filename, {'data': data})

#train_data =
#train_label = 
