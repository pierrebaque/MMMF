
# coding: utf-8

# In[1]:

import matplotlib
matplotlib.use("nbagg")
import math
import numpy.linalg as la
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import cv2
import scipy
import scipy.io as sio
import re, os, glob, pickle, shutil
from shutil import *
from combi import *

from TRW import *
from MF import *
from data_synthetic import *


# ### Clamping with HO

# In[2]:

np.argsort(np.arange(5))[::-1]
print np.max(np.arange(5))


# In[3]:

import copy
clamp_one = 1.0-1.0e-12

def fusion_clamps_HO(clamps,HOs):
    fused_clamps = clamps
    for X in HOs:
        for i in range(X[0].shape[0]):
            fused_clamps.append((X[0][i],X[1][i]*clamp_one)) 
    return fused_clamps

def add_clamplist_random_top(intial_clamplist,intial_HOlist,N_vars,index):
    print 'random N_vars %d'%N_vars
    count =0
    variable_to_clamp = random.randint(0,N_vars-1)
    #Look for a free variable to clamp
    #print 'initial_clamplist',intial_clamplist
    if len(intial_clamplist)>0:
        clamps = intial_clamplist[index]
        HOs = intial_HOlist[index]
        fused_clamps = fusion_clamps_HO(clamps,HOs)

        while len(np.where(np.asarray(fused_clamps)[:,0]==variable_to_clamp)[0] )!=0:
            variable_to_clamp = (variable_to_clamp+1)%N_vars
            count +=1
            if count == N_vars+1:
                print 'no more variable to clamp'
                raise Exception('no more variable to clamp')
                return

    if variable_to_clamp > N_vars -1 :
        print 'error, clamping variable %d in random_top'%variable_to_clamp
    clamplist = copy.copy(intial_clamplist)
    l = len(clamplist)
    if l ==0:
        clamplist=[[(variable_to_clamp,clamp_one)],[(variable_to_clamp,-clamp_one)]]
    else:
        current_clamps = copy.copy(clamplist[index])
        del clamplist[index]
        clamplist.insert(index,current_clamps + [(variable_to_clamp,clamp_one)])
        clamplist.insert(index+1,current_clamps + [(variable_to_clamp,-clamp_one)])
    return clamplist

def add_clamplist_HO_transition_top(A,B,initial_clamplist,initial_HOlist,index,style = 'absolute',order = 1, T_gap = 1.2,threshold = 0.05,len_threshold=4,T_init_search = 1,MF_step = 0.3):
    max_size_HO =order
    variable_to_clamp = -1
    if len(initial_clamplist)>0:
        clamps = initial_clamplist[index]
        HOs=initial_HOlist[index]
    else:
        clamps =[]
        HOs =[]
        
    count =0
    stop = False
    stopped_no_trans = False
    T = T_init_search
    N_vars = B.shape[0]
    x_1 = meanfield_parallel_natural_HO(A,B, MF_step, 10000,clamps,HOs)[0]

    while stop == False:
        T = T*T_gap
        x = meanfield_parallel_natural_HO(A/T,B/T, MF_step, 10000,clamps,HOs)[0]
        #Preselect variables
        if style == 'domke' or  style == 'random' or style == 'singleton':
            #Just check that not clamped
            #These method therefore only work for order 1.
            fused_clamps = clamps +[(X[0][0],X[1][0]*clamp_one) for X in HOs]
            transition = (np.ones(N_vars))
            for (variable,value) in fused_clamps:
                transition[variable] =0
            transition = transition > 0
            #print transition
        elif style == 'absolute' or style == 'openabsolute':
            transition = np.logical_or((x < threshold) * (x_1 > 0.9),(x > -threshold) * (x_1 < -0.9))
        else:
            transition = (np.abs(x) < threshold) * (np.abs(x_1) > 0.9)
            
        where = np.where(transition)[0]
        #Select them using either random or max weights
        if len(where)>len_threshold:
            stop = True
            W_score = 0*B - 1e4
            for i in where:
                W_score[i] = np.reshape(np.abs(A[:,i]),(N_vars)).dot(transition)
                
            if style == 'domke' or style == 'open' or style == 'openabsolute':
                sorted_indices = np.argsort(W_score)[::-1] #From largest W_score
                sorted_indices = sorted_indices[0:max_size_HO]
                where = sorted_indices

            else:
                perm = np.random.permutation(len(where))[0:max_size_HO]
                where = where[perm]
            #print 'where',where
            variables_to_clamp = where
            values_to_clamp = np.ones(where.shape)
            #print x
            values_to_clamp[x[where]<0] = -1 
#             print 'found transition',W_score, variable_to_clamp
        
        else:

            if count>30:
                #print 'count out with T',T
                stop = True
                stopped_no_trans = True
            else:
                count += 1
                
    #Taking care of clamps
    clamplist = copy.copy(initial_clamplist)
    HOlist = copy.copy(initial_HOlist)
    
    if stopped_no_trans:
        clamplist = add_clamplist_random_top(initial_clamplist,HOlist,N_vars,index)
        HOlist.insert(index+1,[] )
        
    else:
        if np.max(variables_to_clamp)> N_vars-1:
            print 'error clamping variable %d, style %s'%(np.max(variables_to_clamp),style)
        
        #Variables to clamp on left branch
        left_list =[]
        for i,v in enumerate(variables_to_clamp):
            left_list.append((v,values_to_clamp[i]*clamp_one))

        l = len(clamplist)
        if l ==0:
            clamplist=[left_list,[]]
        else:
            current_clamps = copy.copy(clamplist[index])
            del clamplist[index]
            clamplist.insert(index,current_clamps + left_list)
            clamplist.insert(index+1,current_clamps )
        
        if(len(variables_to_clamp) <0): # remove this for the moment...
            #Variables to clamp on right branch if one value to clamp
            right_list =[(variables_to_clamp[0],-1*values_to_clamp[0]*clamp_one)]

            current_clamps = copy.copy(clamplist[index+1])
            clamplist[index+1] = current_clamps + right_list

        else:
            #Taking care of higher orders

            HOlist = copy.copy(initial_HOlist)
            new_HO_var =[]
            new_HO_val =[]
            for i,v in enumerate(variables_to_clamp):
                new_HO_var.append(v)
                new_HO_val.append(-1*values_to_clamp[i])
            new_HO = (np.asarray(new_HO_var),np.asarray(new_HO_val),500)
            #l should be the same for HOlist and clamplist
            if l ==0:
                HOlist=[[],[new_HO]]
            else:

                current_HOs = copy.copy(HOlist[index])

                current_HOs.append(new_HO)
                HOlist.insert(index+1,current_HOs )

    #print "end add clamplist"

        
    return clamplist,HOlist


# In[4]:

def tree_clamping(A,B,n_nodes = 100,verbose = False,history = True,order = 1, T_gap = 1.1,T_init_search = 1,len_threshold =3,threshold = 0.05,style = 'random',MF_step = 0.3,log_norm = 0):
    clamplist =[]
    HOlist =[[]]
    Z_MF_list = []

    clamps =[]
    HOs =[]
    Z_MF_list_h=[]


    N = A.shape[0]
    index = 0 

    clamplist_prev = []
    HOlist_prev = []
    gap_prev = 1e2000

    Z_MMF =0
    
    if n_nodes==0:
        clamplist =[[]]
        Z_MF_list = [get_Z_MF(A,B,[])]

        
    index = 0 
    for d in range(0,n_nodes):
        #print 'node %d'%d
        #add new clamps
        if len(Z_MF_list)>0:
            #For smart gap based
            #index = np.argmax(np.asarray(Z_TRW_list) - np.asarray(Z_MF_list))
            #For breadth first
            index = (index +2)%(len(Z_MF_list))

        #Add this node to the clamplist
        
        #Clamp
        #clamplist = add_clamplist_random_top(clamplist,N,index)

        clamplist,HOlist = add_clamplist_HO_transition_top(A,B,clamplist,HOlist,index,style =style,order = order, T_gap = T_gap,threshold = threshold ,len_threshold=len_threshold,T_init_search = T_init_search )
        #print clamplist
        #Update Z_MF_list
        #print 'index',index,'len',len(Z_MF_list)
        if len(Z_MF_list)>0:
            del Z_MF_list[index]
        Z_MF = get_Z_MF(A,B,clamplist[index], HOlist[index],step = MF_step,log_norm = log_norm)
        Z_MF_list.insert(index,Z_MF)
        #print 'HOlist',HOlist
        #print 'HOList'
        Z_MF = get_Z_MF(A,B,clamplist[index+1], HOlist[index+1],step = MF_step,log_norm = log_norm)
        Z_MF_list.insert(index+1,Z_MF)

        
        #print Z_MF_list
        #Update Z_TRW_list
        
        if verbose:

            print 'Z_MMF : %e'%np.sum(np.asarray(Z_MF_list))
#          print 'Z_MTRW : %e'%np.sum(np.asarray(Z_TRW_list))
            print '################### Gap : %e'%np.sum(np.asarray(Z_TRW_list) - np.asarray(Z_MF_list))
        if history:
            Z_MF_list_h.append(copy.copy(Z_MF_list))
            #Z_TRW_list_h.append(copy.copy(Z_TRW_list))
            

        clamplist_prev = clamplist
        
    if history:
        return clamplist, Z_MF_list_h,HOlist
    else:
        return clamplist, Z_MF_list,HOlist
 


# In[5]:


# ## Run all experiments

# Grid attractive

# In[6]:

def save_Z(Z_MF_list_h,path):
    Z_MMFs =[]

    for nodes in range(0,len(Z_MF_list_h)):
        Z_MMFs.append(np.log(np.sum(np.asarray(Z_MF_list_h[nodes]))))

    Z_MMFs_T_BF = copy.copy(Z_MMFs)
    Ours= Z_MMFs_T_BF

    pickle.dump(Ours,open(path,'wb'))
    

def list_fusion(clamplist,HOlist):
    new_list = []
    for i in range(len(clamplist)):
        new_list.append(clamplist[i] +[(X[0][0],X[1][0]*clamp_one) for X in HOlist[i]])
        
    return new_list

def get_Z_TRW(A,B,clamps,log_norm = 0):
    x_approx,Z_TRW_loc = TRW(A,B,10,clamps, anneal=False, verbose=False,log_norm = log_norm)
    
    return Z_TRW_loc

def Q_inter(q_1,q_2):
    return np.prod(q_1*q_2 + (1-q_1)*(1-q_2))

def check_disjointness(A,B,clamplist,HOlist,(method,order,ex)):
    return 
    x_list = []
    for i in range(len(clamplist)):
        x_list.append(meanfield_parallel_natural_HO(A,B, 0.4, 10000,clamplist[i],HOlist[i])[0])

    for i in range(len(x_list)):
        for j in range(i):
            q_1 = (x_list[i]+1)/2
            q_2 = (x_list[j]+1)/2
            Qint =  Q_inter(q_1,q_2)
            if Qint > 1e-5:
                print 'intesection warning Qint = %f for %s %d %d'%(Qint,method,order,ex)


# In[8]:

#Bulk


def run_experiment_ours(ex,data = 'ga',N = 7):
    
    #Same for all
    radius = 1
    n_nodes = 200
    general_name = 'clamping_HO_final_3_%s_N%d_'
    name = general_name%(data,N)
    log_norm = 0
    #Create data
    if data == 'gm':
        (A,B) = generate_grid_format(N,radius,1,1)[0:2]    
        A = 3.0*A
        B = 2.0*B
        if N >12:
            log_norm = 400
        
    if data == 'rm':
        (A,B) = generate_random(n_vars = N*N,density = 0.08,T = 1, link1 = False)  
        A = 3.0*A
        B = 2.0*B
        if N >12:
            log_norm = 800
    
    if data == 'ga':
        (A,B) = generate_grid_format(N,radius,1,1)[0:2]    
        A = 3.0*A
        B = 2.0*B
        A = np.abs(A)
        if N >12:
            log_norm = 500

    if data == 'ra':
        (A,B) = generate_random(n_vars = N*N,density = 0.08,T = 1, link1 = False)  
        A = 3.0*A
        B = 2.0*B
        A = np.abs(A)
        if N >12:
            log_norm = 1500
        
    
    #Instruct parameters
    if data =='ra':
        threshold = 0.08
        T_init_search = 7
        T_gap = 1.04
        len_threshold=3
        
    if data == 'gm':
        threshold = 0.05
        T_init_search = 1
        T_gap = 1.2
        len_threshold=3
        
    if data == 'rm':
        threshold = 0.05
        T_init_search = 1
        T_gap = 1.2
        len_threshold=3

    
    if data == 'ga':
        threshold = 0.05
        T_init_search = 1
        T_gap = 1.2
        len_threshold=3


        
    
    #ours order 1
    method = 'ours'
    order = 1

    clamplist, Z_MF_list_h,HOlist = tree_clamping(A,B,n_nodes = n_nodes,order = order,style = 'absolute',threshold = threshold, T_gap = T_gap,T_init_search = T_init_search, len_threshold=len_threshold,log_norm = log_norm)
    save_Z(Z_MF_list_h,'./experiments/clamping/'+name+'%s_%d_%04d.pickle'%(method,order,ex))
    check_disjointness(A,B,clamplist,HOlist,(method,order,ex))
    print 'OK %s %d %d'%(method,order,ex)
    
    #ours order 2
    method = 'ours'
    order = 2 
    clamplist, Z_MF_list_h,HOlist = tree_clamping(A,B,n_nodes = n_nodes,order = order,style = 'absolute',threshold = threshold, T_gap = T_gap,T_init_search = T_init_search, len_threshold=len_threshold,log_norm = log_norm)
    check_disjointness(A,B,clamplist,HOlist,(method,order,ex))
    save_Z(Z_MF_list_h,'./experiments/clamping/'+name+'%s_%d_%04d.pickle'%(method,order,ex))
    print 'OK %s %d %d'%(method,order,ex)
    
    #ours order 3
    method = 'ours'
    order = 3 
    clamplist, Z_MF_list_h,HOlist = tree_clamping(A,B,n_nodes = n_nodes,order = order,style = 'absolute',threshold = threshold, T_gap = T_gap,T_init_search = T_init_search, len_threshold=len_threshold,log_norm = log_norm)
    check_disjointness(A,B,clamplist,HOlist,(method,order,ex))
    save_Z(Z_MF_list_h,'./experiments/clamping/'+name+'%s_%d_%04d.pickle'%(method,order,ex))
    print 'OK %s %d %d'%(method,order,ex)
    
#     #ours open order 1
    method = 'ours_open'
    order = 1 
    clamplist, Z_MF_list_h,HOlist = tree_clamping(A,B,n_nodes = n_nodes,order = order,style = 'openabsolute',threshold = threshold, T_gap = T_gap,T_init_search = T_init_search, len_threshold=len_threshold,log_norm = log_norm)
    check_disjointness(A,B,clamplist,HOlist,(method,order,ex))
    save_Z(Z_MF_list_h,'./experiments/clamping/'+name+'%s_%d_%04d.pickle'%(method,order,ex))
    print 'OK %s %d %d'%(method,order,ex)
    
    #####Save TRW bounds from this method
    new_clamplist = list_fusion(clamplist,HOlist)
    #print new_clamplist
    Z_TRW_list = [get_Z_TRW(A,B,clamps,log_norm = log_norm) for clamps in new_clamplist]
    TRW_bound = np.log(np.sum(Z_TRW_list))
    pickle.dump(TRW_bound,open('./experiments/clamping/'+name+'TRW_bound_%04d.pickle'%ex,'wb'))
    #####
    print 'OK TRW %s %d %d'%(method,order,ex)
    
    #ours open order 2
    method = 'ours_open'
    order = 2
    clamplist, Z_MF_list_h,HOlist = tree_clamping(A,B,n_nodes = n_nodes,order = order,style = 'openabsolute',threshold = threshold, T_gap = T_gap,T_init_search = T_init_search, len_threshold=len_threshold,log_norm = log_norm)
    check_disjointness(A,B,clamplist,HOlist,(method,order,ex))
    save_Z(Z_MF_list_h,'./experiments/clamping/'+name+'%s_%d_%04d.pickle'%(method,order,ex))
    print 'OK %s %d %d'%(method,order,ex)
    
    #ours open order 3
    method = 'ours_open'
    order = 3 
    clamplist, Z_MF_list_h,HOlist = tree_clamping(A,B,n_nodes = n_nodes,order = order,style = 'openabsolute',threshold = threshold, T_gap = T_gap,T_init_search = T_init_search, len_threshold=len_threshold,log_norm = log_norm)
    check_disjointness(A,B,clamplist,HOlist,(method,order,ex))
    save_Z(Z_MF_list_h,'./experiments/clamping/'+name+'%s_%d_%04d.pickle'%(method,order,ex))
    print 'OK %s %d %d'%(method,order,ex)
    
    #domke order 1
    method = 'domke'
    order = 1 
    clamplist, Z_MF_list_h,HOlist = tree_clamping(A,B,n_nodes = n_nodes,order = order,style = 'domke',threshold = threshold, T_gap = T_gap,T_init_search = T_init_search, len_threshold=len_threshold,log_norm = log_norm)
    check_disjointness(A,B,clamplist,HOlist,(method,order,ex))
    save_Z(Z_MF_list_h,'./experiments/clamping/'+name+'%s_%d_%04d.pickle'%(method,order,ex))
    print 'OK %s %d %d'%(method,order,ex)
    
    
    #random order 1
    method = 'random'
    order = 1
    clamplist, Z_MF_list_h,HOlist = tree_clamping(A,B,n_nodes = n_nodes,order = order,style = 'random',threshold = threshold, T_gap = T_gap,T_init_search = T_init_search, len_threshold=len_threshold,log_norm = log_norm)
    check_disjointness(A,B,clamplist,HOlist,(method,order,ex))
    save_Z(Z_MF_list_h,'./experiments/clamping/'+name+'%s_%d_%04d.pickle'%(method,order,ex))
    print 'OK %s %d %d'%(method,order,ex)



# In[ ]:

n_examples = 3*33

from joblib import Parallel, delayed
import multiprocessing

n_threads =33

for block in range(0,n_examples/n_threads+1):
    for N in [13]:
        for data in ['ga','gm','ra','rm']:
            print 'block %d, data %s, N%d'%(block,data,N)
            local_list = range(n_threads*block,n_threads*(block+1) )
            Parallel(n_jobs=n_threads)(delayed(run_experiment_ours)(ex,data = data, N = N) for ex in local_list)





# In[ ]:



