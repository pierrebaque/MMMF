
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

def meanfield_parallel_natural(A,B, step, n_iterations,clamps,init_factor = 1.0, anneal=False, verbose=False,out_history = False,check_gradient = True):

    N = A.shape[0]
    indices = np.arange(0,N) 
    x_history = []

    x = (2.0 * np.random.rand(N) - 1.0)*init_factor
    #clamping
    
    for i in range(0,len(clamps)):
        x[clamps[i][0]] = clamps[i][1]
    
    x_nat = np.log((x + 1) / 2)

    for it in range(n_iterations):
        x_nat_up = np.zeros_like(x)        
        # updating ALL the variables at the same time
        for i in indices:
            x_nat_up[i] = x.dot(A[:,i]) + B[i]
            
        # deterministic annealing    
        T =1.0# 1.0 / (it+1) if anneal else 0.1
            
        x_nat = step * x_nat_up + (1.0 - step) * x_nat
        x = 2.0 * scipy.special.expit(2.0 * x_nat / T) - 1.0
        
        #clamping variables
        for i in range(0,len(clamps)):
            x[clamps[i][0]] = clamps[i][1]
            
        x_history.append(x)
        
        
        if verbose and it % 1 == 0:
             print it, compute_KL(A,B, x)
                
        if it%1 ==0 and check_gradient:
            grad_norm = np.sqrt(np.mean(compute_gradient(A,B, x,x_nat,clamps)**2))
            if verbose:
                print 'grad_norm', grad_norm
            if grad_norm < 1e-6:
                if out_history:
                    return x, x_nat,x_history
                else:
                    return x, x_nat
                
    print 'reached end with grad', grad_norm
    
    if out_history:
        return x, x_nat,x_history
    else:
        return x, x_nat
    
def meanfield_sweep(A,B, step, n_iterations,clamps,init_factor = 1.0, anneal=False, verbose=False,out_history = False,check_gradient = True):

    N = A.shape[0]
    indices = np.arange(0,N) 
    x_history = []

    x = (2.0 * np.random.rand(N) - 1.0)*init_factor
    #clamping
    
    for i in range(0,len(clamps)):
        x[clamps[i][0]] = clamps[i][1]
    
    x_nat = np.log((x + 1) / 2)

    for it in range(n_iterations):
        x_nat_up = np.zeros_like(x)        
        # updating ALL the variables at the same time
        for i in indices:
            x_nat_up[i] = x.dot(A[:,i]) + B[i]
    
            x[i] = 2.0 * scipy.special.expit(2.0 * x_nat_up[i] / T) - 1.0
        
        #clamping variables
        for i in range(0,len(clamps)):
            x[clamps[i][0]] = clamps[i][1]
            
        x_history.append(x)
        
        
        if verbose and it % 1 == 0:
             print it, compute_KL(A,B, x)
                
        if it%1 ==0 and check_gradient:
            grad_norm = np.sqrt(np.mean(compute_gradient(A,B, x,x_nat,clamps)**2))
            if verbose:
                print 'grad_norm', grad_norm
            if grad_norm < 1e-6:
                if out_history:
                    return x, x_nat,x_history
                else:
                    return x, x_nat
                
    print 'reached end with grad', grad_norm
    
    if out_history:
        return x, x_nat,x_history
    else:
        return x, x_nat

    
def meanfield_parallel_natural_adam(A,B, step, n_iterations,clamps, anneal=False, verbose=False,out_history = False,check_gradient = True,g1 = 0.01,g2 = 0.001):

    N = A.shape[0]
    indices = np.arange(0,N) 
    x_history = []
    d = np.zeros(N)+2.0
    m = np.zeros(N) + 0.0
    v = np.zeros(N) + 2.0
    
    x = 2.0 * np.random.rand(N) - 1.0
    #clamping
    
    for i in range(0,len(clamps)):
        x[clamps[i][0]] = clamps[i][1]
    
    x_nat = np.log((x + 1) / 2)

    for it in range(n_iterations):
        
        x_nat_up = np.zeros_like(x)        
        # updating ALL the variables at the same time
        for i in indices:
            x_nat_up[i] = x.dot(A[:,i]) + B[i]
            
        # deterministic annealing    
        T =1.0# 1.0 / (it+1) if anneal else 0.1
        
        m = (1-g1)*m + g1*x_nat_up
        v = (1-g2)*v + g2*(x_nat_up - x_nat)**2
        d = np.sqrt(v)*2 + 1e-7 #- 1
        step = 1/(1+d)  
        x_nat = step * m + (1.0 - step) * x_nat
        x = 2.0 * scipy.special.expit(2.0 * x_nat / T) - 1.0
        
        
        
        #clamping variables
        for i in range(0,len(clamps)):
            x[clamps[i][0]] = clamps[i][1]
            
        x_history.append(x)
        
        
        if verbose and it % 1 == 0:
             print it, compute_KL(A,B, x)
                
        if it%1 ==0 and check_gradient:
            grad_norm = np.sqrt(np.mean(compute_gradient(A,B, x,x_nat,clamps)**2))
            if verbose:
                print 'grad_norm', grad_norm
            if grad_norm < 5e-4:
                if out_history:
                    return x, x_nat,x_history
                else:
                    return x, x_nat
                
    print 'reached end with grad', grad_norm
    
    if out_history:
        return x, x_nat,x_history
    else:
        return x, x_nat
    

def meanfield_parallel_natural_HO(A,B, step, n_iterations,clamps,HO_potentials, init_factor = 0.001,anneal=False, verbose=False,out_history = False,check_gradient = True):

    N = A.shape[0]
    indices = np.arange(0,N) 
    x_history = []
    Phi_H0 =10

    x = (2.0 * np.random.rand(N) - 1.0)*init_factor
    #clamping
    
    for i in range(0,len(clamps)):

        x[clamps[i][0]] = clamps[i][1]
    
    x_nat = np.log((x + 1) / 2)
    

    for it in range(n_iterations):
        x_nat_up = np.zeros_like(x)        
        # updating ALL the variables at the same time
        for i in indices:
            x_nat_up[i] = x.dot(A[:,i]) + B[i]
            
        # Handling Higher order potentials
        #print 'Len HO',len(HO_potentials)
        max_E_H = 0
        for ho_potential in HO_potentials:
            variables = ho_potential[0]
            #C is the value that has to be taken by at least one of the variables
            C = np.asarray(ho_potential[1])
            #Expectancy of higher order under Q
            E_H = np.prod(0.5*(1 - C*x[variables]))
            E_H_m = 0.5*(C+1)*E_H/(0.5*(1 - C*x[variables]))
            E_H_p = 0.5*(1-C)*E_H/(0.5*(1 - C*x[variables]))
            #print 'before',x_nat_up
            #print 'x_nat_up[variables]',x_nat_up[variables]
            x_nat_up[variables] += Phi_H0*(E_H_m - E_H_p)
            max_E_H = max(max_E_H,E_H)
            #print 'after',x_nat_up
            
        # deterministic annealing    
        T =1.0# 1.0 / (it+1) if anneal else 0.1
            
        x_nat = step * x_nat_up + (1.0 - step) * x_nat
        x = np.clip(2.0 * scipy.special.expit(2.0 * x_nat / T) - 1.0,-0.99999,0.99999)
        
        #clamping variables
        for i in range(0,len(clamps)):
            x[clamps[i][0]] = clamps[i][1]
            
        x_history.append(x)
        
        
        if verbose and it % 1 == 0:
             print it, compute_KL(A,B, x)
                
        if it%1 ==0 and check_gradient:
            #grad_norm = np.sqrt(np.mean(compute_gradient(A,B, x,x_nat,clamps)**2))
            grad_norm = np.sqrt(np.mean((x_nat_up-x_nat)**2))
            if verbose:
                print 'grad_norm', grad_norm
            if grad_norm < 1e-6 and max_E_H <0.00001:
                if out_history:
                    return x, x_nat,x_history
                else:
                    return x, x_nat
            elif grad_norm < 1e-6:
                Phi_H0 = 2*Phi_H0
                
    print 'reached end with grad', grad_norm
    
    if out_history:
        return x, x_nat,x_history
    else:
        return x, x_nat

    
def get_Z_MF(A,B,clamps,HOs,step = 0.3, log_norm = 0):
    x_approx = meanfield_parallel_natural_HO(A,B, step, 40000,clamps,HOs)[0]
    Z_MF_loc = np.exp(-compute_KL(A,B,x_approx) - log_norm)
    
    return Z_MF_loc


def normalize_sign(x):
    return x if (x[0] > 0) else -x


def compute_score(A,B, x):
    return x.T.dot(A).dot(x) + B.dot(x)


def compute_KL(A,B,x):
    p=(x+1)/2

    return -x.T.dot(A).dot(x)/2.0 -B.dot(x)  + p.dot(np.log(np.maximum(p,1e-30))) + (1-p).dot(np.log(np.maximum(1-p,1e-30)))

def compute_gradient(A,B, x,x_nat,clamps):
    p=(x+1)/2.0
    #return -2*A.T.dot(x) -B  +(log(np.maximum(p,1e-30)))/2.0 - (log(np.maximum(1-p,1e-30)))/2.0
    #return -x_nat  +(log(np.maximum(p,1e-30)))/2.0 - (log(np.maximum(1-p,1e-30)))/2.0
    #equivalent but numerically stable 
    grad = -A.T.dot(x) -B  +x_nat
    for i in range(0,len(clamps)):
        grad[clamps[i][0]] = 0

    return grad
