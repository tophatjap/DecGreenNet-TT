#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import math
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class poisson2d:
    def __init__(self, a ):
        self.a = a

    def interior_value(self,x,y):
        return self.a*x*(x-1)*y*(y-1)           

    def boundary_value(self,x,y):
        return 0


class mesh_generator:
    def __init__(self, X_interior , boundary ):
        self.X_interior = X_interior
        self.boundary = boundary

    def generate_random_points(self, ranges, n_train,n_test,n_val,fname):
        n_total = n_train + n_test + n_val
        num_ranges = len(ranges)
        
        train_data = np.random.random_sample([n_total,num_ranges])
        
        n_arrray = np.zeros([n_total,num_ranges+1])
        
        dist_list = list() 
        for i in range(num_ranges):
            dist_list.append(ranges[i][1] - ranges[i][0])
        
        idx = 0
        for i in range(n_total):
            idx = idx+1
            x = train_data[idx-1,:]
            #print(x,y)
            n_arrray[idx-1,0] = idx 
            for j in range(num_ranges):
                n_arrray[idx-1,j+1] =  ranges[j][0] + dist_list[j]*x[j]

        '''                 

            
        for i in range(n_test):
            idx = idx+1
            x,y = train_data[idx-1,:]
            n_arrray[idx-1,0] = idx 
            n_arrray[idx-1,1] = x 
            n_arrray[idx-1,2] = y 

            
        for i in range(n_val):
            idx = idx+1
            x,y = train_data[idx-1,:]
            n_arrray[idx-1,0] = idx 
            n_arrray[idx-1,1] = x 
            n_arrray[idx-1,2] = y 
        '''

        print(n_arrray)            
        x_df = pd.DataFrame(n_arrray)
        x_df.to_csv(fname)
        
    def generate_random_boundary_points(self, ranges, b_range , n_train,n_test,n_val,fname):
        n_total = n_train + n_test + n_val
        num_ranges = len(ranges)
        
        train_data = np.random.random_sample([n_total,num_ranges])
        
        n_arrray = np.zeros([n_total,num_ranges+1])
        
        dist_list = list() 
        for i in range(num_ranges):
            dist_list.append(ranges[i][1] - ranges[i][0])
        
        idx = 0
        for i in range(n_total):
            idx = idx+1
            x = train_data[idx-1,:]
            #print(x,y)
            n_arrray[idx-1,0] = idx 
            for j in range(num_ranges):
                n_arrray[idx-1,j+1] =  ranges[j][0] + dist_list[j]*max([b_range[j]-1,0]) + dist_list[j]*x[j]*(b_range[j]%2)
        
        print(n_arrray)
        x_df = pd.DataFrame(n_arrray)
        x_df.to_csv(fname)
    
    '''
    def generate_random(self,equation, n_train,n_test,n_val,fname):
        n_total = n_train + n_test + n_val
        
        train_data = np.random.random_sample([n_total,2])
        
        n_arrray = np.zeros([n_total,4])
        
        idx = 0
        for i in range(n_train):
            idx = idx+1
            x,y = train_data[idx-1,:]
            print(x,y)
            n_arrray[idx-1,0] = idx 
            n_arrray[idx-1,1] = x 
            n_arrray[idx-1,2] = y 
            n_arrray[idx-1,3] = equation.interior_value(x,y)
            
        for i in range(n_test):
            idx = idx+1
            x,y = train_data[idx-1,:]
            n_arrray[idx-1,0] = idx 
            n_arrray[idx-1,1] = x 
            n_arrray[idx-1,2] = y 
            n_arrray[idx-1,3] = equation.interior_value(x,y)
            
        for i in range(n_val):
            idx = idx+1
            x,y = train_data[idx-1,:]
            n_arrray[idx-1,0] = idx 
            n_arrray[idx-1,1] = x 
            n_arrray[idx-1,2] = y 
            n_arrray[idx-1,3] = equation.interior_value(x,y)

        print(n_arrray)            
        x_df = pd.DataFrame(n_arrray)
        x_df.to_csv(fname)
    ''' 
'''    
class Interior_dataset(Dataset):

    def __init__(self, csv_file,n_train,n_test,n_val,data_type='train'):
        self.data = pd.read_csv(csv_file)
        self.n_train = n_train
        self.n_test = n_test
        self.n_val = n_val
        if data_type=='train':
            self.data = self.data.iloc[:n_train, 1:]
        elif data_type=='test':
            self.data = self.data.iloc[n_train:n_train+n_test, 1:]
        elif data_type=='val':
            self.data = self.data.iloc[n_train+n_test:, 1:]
        self.data  =  self.data[1:]
        print(len(self.data))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        landmarks = self.data.iloc[idx, 1:]
        print(landmarks[1:])
        landmarks = np.array([landmarks])
        sample = landmarks.astype('float')
        #sample = landmarks.astype('float').reshape(-1, 2)

        return sample
'''

class Data_pliter:
    def __init__(self, fname, batch_size ):
        self.batch_size = batch_size

    def getBatch(self):
        
         





        
if __name__ == '__main__':
    e = poisson2d(5)
    m = mesh_generator([0,1],[0])
    m.generate_random_points([[0,1],[0,1]],100,50,30,'poi2d.csv')
    m.generate_random_boundary_points([[0,1],[0,1]], [1,2] ,100,50,30,'b_poi2d.csv')
    
    '''
    datas = Interior_dataset('poi2d.csv', 100,50,30,'val')
    dataloader = DataLoader(datas, batch_size=4,shuffle=True)
    #print(dataloader.len())
    en = enumerate(dataloader)
    print(datas.__getitem__([1,2]))
    '''
    '''
    for batch, x in enumerate(dataloader):
        print(batch)
        print(x)
    '''
