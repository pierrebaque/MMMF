
import math
import numpy.linalg as la
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import scipy
import scipy.io as sio
import re, os, glob, pickle, shutil
from shutil import *
from combi import *

import random

T_sample = 1


def set_value(A,i,j,v):
    A[i,j] = v
    A[j,i] = v
    return

#Energy is exp(-aij xi xj)
def generate_grid_2x2():
    A = np.zeros((4,4))
    set_value(A,0,1,-0.2)
    #set_value(A,1,2,2.0)
    set_value(A,2,3,-0.4)
    #set_value(A,3,0,1.0)
    
    return A

def generate_duo():
    A = np.zeros((2,2))
    set_value(A,0,1,-0.2)
#     set_value(A,1,2,2.0)
#     set_value(A,2,3,-0.4)
    #set_value(A,3,0,1.0)
    
    return A


def generate_random(n_vars = 20,density = 0.4,T = 1, link1 = False):
    A = np.zeros((n_vars,n_vars))
    for i in range(0,n_vars):
        for j in range(0,i):
            if j == 0 and link1:
                r =0
            else:
                r= random.random()

            if(r < density):
                draw = random.random()
                set_value(A,i,j,2*(draw - 0.5)/T)
                
        #Add unary potentials
    B = np.zeros(n_vars)
    for k in range(0,n_vars):
        b  =random.uniform(-1,1)
        B[k] = b
                    

    
    return A,B

#Synthetic grid N*N
def generate_grid_format(N,r,A_scale,B_scale):
    
    A= np.zeros((N*N,N*N))
    B = np.zeros(N*N)
    shift = np.random.rand(1)
    v_factors = []
    f_vars=[]
    f_tables=[]
    v_cards = []
    #Cardinals
    for k in range(0,N*N):
        v_cards.append(2)
        v_factors.append([])
        
    n_factor = 0 
    #Add unary potentials
    for k in range(0,N*N):
        b  =B_scale * random.uniform(-1,1)
        B[k] = b
        v_factors[k].append(n_factor)
        f_vars.append([k])
        f_tables.append([-b,b])
        n_factor = n_factor+1
    
    #Add pairwise potentials
    
    for k in range(0,N*N):
        i_1 = k/N
        j_1 = k%N
        A[k,k] = 0
        for l in range(k,N*N):
            i_2 = l/N
            j_2 = l%N

            if (abs(i_1 - i_2)+abs(j_1 - j_2)<r+1 and abs(i_1 - i_2)+abs(j_1 - j_2)!=0):
                #A[k,l] = np.random.rand(1)-2*(k*1.0)/(N*N*1.0)
                a = A_scale * random.uniform(-1,1)#+8.0
                A[k,l] = a
                A[l,k] = A[k,l]
                
                v_factors[k].append(n_factor)
                v_factors[l].append(n_factor)
                f_vars.append([k,l])
                f_tables.append([a,-a,-a,a])
                n_factor = n_factor+1
                
    return A, B,v_factors, v_cards, f_vars, f_tables





def bruteforce_partition(A,B):
    n_v = A.shape[0]
    a = [-1,1]
    m=[]
    for i in range(0,n_v):
        m.append(a)
        
    m = np.asarray(m)
    
    States = ProductSpace(m)
    
    Z = 0
    
    for m in States:
        x = np.asarray(m)
        Z += np.exp((x.T.dot(A).dot(x)/2.0 + B.dot(x))  )
    
    return Z,States

def sample(Z,States,A,B):
    r= random.random()
    for m in States:
        x = np.asarray(m)
        value = np.exp((x.T.dot(A).dot(x)/2.0 + B.dot(x))/T_sample )/Z

        if value >=r:
            return m
        
        else:
            r -=value
    
    return
