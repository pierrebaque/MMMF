
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

def TRW(A,B,n_iterations,clamps, anneal=False, verbose=False,log_norm = 0):
    w = tree_weights_grid(A)
    N = A.shape[0]
    m = 0.0*A
    #initialise messages and neighbourhoods
    Neigh = []
    for i in range(0,N):
        N_i = []
        for j in range(0,N):
            if A[i,j]!=0:
                N_i.append(j)
                m[i,j] = 2*random.random()-1
        Neigh.append(N_i)
                        
    #We iterate over iterations
    for it in range(0,n_iterations):
        #We loop through variables in order and update outgoing messages
        for i in range(0,N):
            m = pass_message(i,m,Neigh[i],w,A,B)
            #we clamp back the message if needed
            if len(clamps)>0:
                if len(np.where(np.asarray(clamps)[:,0]==i)[0] )!=0:
                    index_in_clamps =np.where(np.asarray(clamps)[:,0]==i)[0][0]
                    m = pass_message_clamped(i,m,Neigh[i],w,A,B,clamps[index_in_clamps][1])

            
            
    #Magnetisation
    x = get_magnetisation_vector(m,Neigh,w,A,B,clamps)
    c_x = get_cross_magnetisation_matrix(m,Neigh,w,A,B,clamps,x)
    
    
    #Expected Energy 

    E =  np.sum(c_x*A)/2.0 + B.dot(x)
    Neg_Entropy = - np.sum(get_entropy_vector(x))
    Weighted_Mutual_Info = np.sum(w*get_mutual_info_matrix(m,Neigh,w,A,B,clamps)/2.0)
    
    #print 'E',E,'Neg',Neg_Entropy,'Weight',Weighted_Mutual_Info
    Z = np.exp(E - Neg_Entropy - Weighted_Mutual_Info - log_norm) 

    
    return x,Z
            
            


def pass_message(i,m,N_i,w,A,B):
    z_p_i = 1
    z_m_i = 1
    for v in N_i:
        z_p_i *= ((m[v,i]+1)/2.0)**w[v,i]
        z_m_i *= ((1-m[v,i])/2.0)**w[v,i]
        

    
    for j in N_i:
    
        M_pp = np.exp(B[i]+A[i,j]/w[i,j])*z_p_i/((m[j,i]+1)/2.0)
        M_mp = np.exp(-B[i]-A[i,j]/w[i,j])*z_m_i/((-m[j,i]+1)/2.0)
        M_pm = np.exp(B[i]-A[i,j]/w[i,j])*z_p_i/((m[j,i]+1)/2.0)
        M_mm = np.exp(-B[i]+A[i,j]/w[i,j])*z_m_i/((-m[j,i]+1)/2.0)
        
        m[i,j] =  (M_pp+M_mp-M_pm-M_mm)/(M_pp+M_mp+M_pm+M_mm) 
    
    return m

def get_magnetisation(i,m,N_i,w,A,B):
    z_p = 1
    z_m = 1
    for v in N_i:
        z_p *= ((m[v,i]+1)/2.0)**w[v,i]
        z_m *= ((1-m[v,i])/2.0)**w[v,i]
    
    x_i = (np.exp(B[i])*z_p - np.exp(-B[i])*z_m)/(np.exp(B[i])*z_p + np.exp(-B[i])*z_m)
    
    return x_i

def get_cross_magnetisation(i,j,m,N_i,N_j,w,A,B):
    z_p_i = 1
    z_m_i = 1
    for v in N_i:
        z_p_i *= ((m[v,i]+1)/2.0)**w[v,i]
        z_m_i *= ((1-m[v,i])/2.0)**w[v,i]
        
    z_p_j = 1
    z_m_j = 1
    for v in N_j:
        z_p_j *= ((m[v,j]+1)/2.0)**w[v,j]
        z_m_j *= ((1-m[v,j])/2.0)**w[v,j]
    
    
    
    T_pp = np.exp(B[i]+B[j]+A[i,j]/w[i,j])*z_p_i/((m[j,i]+1)/2.0)*z_p_j/((m[i,j]+1)/2.0) 
    T_mp = np.exp(-B[i]+B[j]-A[i,j]/w[i,j])*z_m_i/((-m[j,i]+1)/2.0)*z_p_j/((m[i,j]+1)/2.0)
    T_pm = np.exp(B[i]-B[j]-A[i,j]/w[i,j])*z_p_i/((m[j,i]+1)/2.0)*z_m_j/((-m[i,j]+1)/2.0)
    T_mm = np.exp(-B[i]-B[j]+A[i,j]/w[i,j])*z_m_i/((-m[j,i]+1)/2.0)*z_m_j/((-m[i,j]+1)/2.0)
    
    normalisation = T_pp+T_mp+T_pm +T_mm
    
    T_pp = T_pp /normalisation
    T_pm = T_pm /normalisation
    T_mp = T_mp /normalisation
    T_mm = T_mm /normalisation
        
    
    c_x_ij = T_pp - T_pm -T_mp +T_mm
    
    return c_x_ij

def get_mutual_information(i,j,m,N_i,N_j,w,A,B):
    z_p_i = 1
    z_m_i = 1
    for v in N_i:
        z_p_i *= ((m[v,i]+1)/2.0)**w[v,i]
        z_m_i *= ((1-m[v,i])/2.0)**w[v,i]
        
    z_p_j = 1
    z_m_j = 1
    for v in N_j:
        z_p_j *= ((m[v,j]+1)/2.0)**w[v,j]
        z_m_j *= ((1-m[v,j])/2.0)**w[v,j]
    
    
    T_pp = np.exp(B[i]+B[j]+A[i,j]/w[i,j])*z_p_i/((m[j,i]+1)/2.0)*z_p_j/((m[i,j]+1)/2.0) 
    T_mp = np.exp(-B[i]+B[j]-A[i,j]/w[i,j])*z_m_i/((-m[j,i]+1)/2.0)*z_p_j/((m[i,j]+1)/2.0)
    T_pm = np.exp(B[i]-B[j]-A[i,j]/w[i,j])*z_p_i/((m[j,i]+1)/2.0)*z_m_j/((-m[i,j]+1)/2.0)
    T_mm = np.exp(-B[i]-B[j]+A[i,j]/w[i,j])*z_m_i/((-m[j,i]+1)/2.0)*z_m_j/((-m[i,j]+1)/2.0)
    
    normalisation = T_pp+T_mp+T_pm +T_mm
    
    T_pp = T_pp /normalisation
    T_pm = T_pm /normalisation
    T_mp = T_mp /normalisation
    T_mm = T_mm /normalisation
    
    I_ij = (T_pp*np.log(T_pp/((T_pp+T_pm)*(T_pp + T_mp))) 
            + T_pm*np.log(T_pm/((T_pp+T_pm)*(T_pm + T_mm))) 
            + T_mp*np.log(T_mp/((T_mp+T_mm)*(T_pp + T_mp))) 
            + T_mm*np.log(T_mm/((T_pm+T_mm)*(T_mp + T_mm))))
    
    return I_ij

    
def get_magnetisation_vector(m,Neigh,w,A,B,clamps):
    x = np.zeros(A.shape[0])
    for i in range(0,A.shape[0]):
        x[i]= get_magnetisation(i,m,Neigh[i],w,A,B)
        
    for (i_clamped,value) in clamps:
        x[i_clamped] = value

        
    return x

def get_cross_magnetisation_matrix(m,Neigh,w,A,B,clamps,x):
    
    #Cross
    c_x = 0.0*A
    index = np.where(A!=0)
    for (i,j) in zip(index[0],index[1]):
        if i < j: 
            cross = get_cross_magnetisation(i,j,m,Neigh[i],Neigh[j],w,A,B)
                
            c_x[i,j] = cross
            c_x[j,i] = cross
            
    for (i_clamped,value) in clamps:
        for j in Neigh[i_clamped]:
            cross = get_cross_magnetisation_clamped(i_clamped,j,m,Neigh[i_clamped],Neigh[j],w,A,B,value)
            c_x[i_clamped,j] =value*x[j]
            c_x[j,i_clamped] =value*x[j]
            
            #print 'cross',cross,'get_magnetisation(i,m,N_i,w,A,B)', value*get_magnetisation(j,m,Neigh[j],w,A,B)

                
    return c_x



def get_mutual_info_matrix(m,Neigh,w,A,B,clamps):
    #Cross
    I = 0.0*A
    index = np.where(A!=0)
    for (i,j) in zip(index[0],index[1]):
        if i < j: 
            mutual = get_mutual_information(i,j,m,Neigh[i],Neigh[j],w,A,B)

            I[i,j] = mutual
            I[j,i] = mutual
            
    for (i_clamped,value) in clamps:
        for j in Neigh[i_clamped]:
            I[i_clamped,j] =0
            I[j,i_clamped] =0


    return I

def get_entropy_vector(x,eps = 1e-8):
    H = 0.0*x
    for i in range(0,H.shape[0]):
        q_i = x[i]/(2.0+eps) + 0.5
        H[i]= -q_i*np.log(q_i) - (1-q_i)*np.log(1-q_i)
        
    return H


## For clamped 

def get_cross_magnetisation_clamped(i,j,m,N_i,N_j,w,A,B,value):
    z_p_i = np.round((value+1)/2.0)
    z_m_i = np.round((1-value)/2.0)
        
    z_p_j = 1
    z_m_j = 1
    for v in N_j:
        if v!=i:
            z_p_j *= ((m[v,j]+1)/2.0)**w[v,j]
            z_m_j *= ((1-m[v,j])/2.0)**w[v,j]
    
    
    
    T_pp = np.exp(B[j])*z_p_i*z_p_j
    T_mp = np.exp(B[j])*z_m_i*z_p_j
    T_pm = np.exp(-B[j])*z_p_i*z_m_j
    T_mm = np.exp(-B[j])*z_m_i*z_m_j
    
    normalisation = T_pp+T_mp+T_pm +T_mm
    
    T_pp = T_pp /normalisation
    T_pm = T_pm /normalisation
    T_mp = T_mp /normalisation
    T_mm = T_mm /normalisation
        
    
    c_x_ij = T_pp - T_pm -T_mp +T_mm
    
    return c_x_ij


def pass_message_clamped(i,m,N_i,w,A,B,value):
    z_p_i = np.round((value+1)/2.0)
    z_m_i = np.round((1-value)/2.0)
        

    
    for j in N_i:
    
        M_pp = np.exp(A[i,j]/w[i,j])*z_p_i
        M_mp = np.exp(-A[i,j]/w[i,j])*z_m_i
        M_pm = np.exp(-A[i,j]/w[i,j])*z_p_i
        M_mm = np.exp(+A[i,j]/w[i,j])*z_m_i
        
        m[i,j] =  (M_pp+M_mp-M_pm-M_mm)/(M_pp+M_mp+M_pm+M_mm) 
    
    return m


def tree_weights_grid(A):
    #From simple calculation, we assume that the weights are all 1/2+1/2N
    w = A*0.0
    N = A.shape[0]
    w[A!=0]= 0.5*(1+1.0/np.sqrt(N))
    
    return w







    
