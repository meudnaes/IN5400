import os,sys,numpy as np

import torch

import time

def forloopdists(feats,protos):

    #YOUR implementation here
    N, D = feats.shape
    P = protos.shape[0]
    
    dists = np.zeros((N, P))
    
    for i in range(N):
        for j in range(P):
            #for k in range(D):
            #    dists[i,j] += (feats[i, k] - protos[j, k])**2
            dists[i,j] = np.sum((feats[i, :] - protos[j, :])**2)
            
    return dists

def numpydists(feats,protos):
    #YOUR implementation here
    dists = ((feats**2).sum(axis=1, keepdims=True) + (protos**2).sum(axis=1)
                    - 2*feats.dot(protos.T))
    return dists
  
def pytorchdists(feats0,protos0,device):
  
    #YOUR implementation here
    feats = torch.tensor(feats0, device = device)
    protos = torch.tensor(protos0, device = device)
    dists = (torch.sum(feats**2, axis=1, keepdims=True)+ torch.sum(protos**2, axis=1)
                    - 2*torch.mm(feats, torch.transpose(protos, 0, 1)))
    return dists.data.numpy()

def run():

    ########
    ##
    ## if you have less than 8 gbyte, then reduce from 250k
    ##
    ###############
    feats=np.random.normal(size=(100000, 300)) #5000 instead of 250k for forloopdists
    protos=np.random.normal(size=(500,300))
    
    '''
    since = time.time()
    dists0=forloopdists(feats,protos)
    time_elapsed=float(time.time()) - float(since)
    print('Comp complete in {:.3f}s'.format( time_elapsed ))
    '''

    device=torch.device('cpu')
    since = time.time()

    dists1=pytorchdists(feats,protos,device)


    time_elapsed=float(time.time()) - float(since)

    print('Comp complete in {:.3f}s'.format( time_elapsed ))
    print(dists1.shape)

    #print('df0',np.max(np.abs(dists1-dists0)))


    since = time.time()

    dists2=numpydists(feats,protos)

    time_elapsed=float(time.time()) - float(since)

    print('Comp complete in {:.3f}s'.format( time_elapsed ))

    print(dists2.shape)

    print('df',np.max(np.abs(dists1 - dists2)))


if __name__=='__main__':
    run()
