#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch.nn as nn
import torch
import math
import numpy as np



def Poisson_XD(x,a):
    sx = x.shape
    y = 1
    for i in range(sx[1]):
        y = y*(x[:,i])*(x[:,i]-1)
    
    return 0.5*a*y


def Poisson_XD_dx(x,a):
    sx = x.shape
    print(sx[1])
    yall = 2*Poisson_XD(x,a)
    y = 0
    
    for i in range(sx[1]):
        #print(x[:,i].shape)
        y = y + yall/(x[:,i]*(x[:,i]-1))

    return y



## Yuankai Teng et al. Learning Green’s Functions of Linear Reaction-Diffusion Equations with Application to Fast Numerical Solver
def Poisson_homogenous(x):
    c = np.sin(2*np.pi*x)
    return np.prod(c,1)

def Poisson_homogenous_dx(x):
    sx =  x.shape
    print(sx[1])
    c = np.sin(2*np.pi*x)
    return -4*(np.pi**2)*np.prod(c,1)*sx[1]

def Poisson_inhomogenous(x):
    c = np.cos(np.pi*x)
    return np.prod(c,1)

def Poisson_inhomogenous_dx(x):
    sx =  x.shape
    #print(sx[1])
    c = np.cos(np.pi*x)
    return -(np.pi**2)*np.prod(c,1)*sx[1]

'''
def Poisson_inhomogenous_dx(x):
    c = -(np.pi**2)*np.cos(np.pi*x)
    return np.prod(c,1)
'''
