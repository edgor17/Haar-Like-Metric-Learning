#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 18:53:25 2023

@author: Evan
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 17:40:37 2023

@author: Evan
"""
import scipy as scipy
import scipy.sparse
import numpy as np
from ete3 import Tree


def dcor(D1,D2):
    n=len(D1)
    H=np.eye(n)-np.ones((n,n))/n
    D1cent=H@D1@H
    D2cent=H@D2@H
    D1var=np.sum(np.multiply(D1cent,D1cent))/(n*n)
    D2var=np.sum(np.multiply(D2cent,D2cent))/(n*n)
    cov=np.sum(np.multiply(D1cent,D2cent))/(n*n)
    return np.sqrt(cov/np.sqrt(D1var*D2var))



def find_matching_index(list1, list2):

    inverse_index = { element: index for index, element in enumerate(list1) }

    return [(index)
        for index, element in enumerate(list2) if element in inverse_index]
       

def Haar_Build(tree):
    '''
    

    Parameters
    ----------
    tree : str
        filepath to .nwk tree, in current version must be bifurcating

    Returns
    -------
    sparsehaarlike : scipy.sparse.csr_matrix
        Haar-like basis

    '''
    t = tree
    node2leaves = t.get_cached_content() #return dictionary of node instances, allows quick access to node attributes without traversing the tree
    numleaves=len(t) 
    sparsehaarlike=scipy.sparse.lil_matrix((numleaves,numleaves)) #Initialize row-based list of lists sparse matrix to store Haar-like vectors
    allleaves=t.get_leaves()
    mastersplit=t.children
    lilmat=scipy.sparse.lil_matrix((numleaves,numleaves)) #this is where we collect lstar vectors
    
    
    i=0 #ordering of nodes in post order traversal
    for node in t.traverse("postorder"):
        node.add_features(pos=find_matching_index(node,allleaves)) #store indices of leaves under each internal node
        veclen=len(node2leaves[node])
        if not node.is_leaf():
            node.add_features(loc=i) #add node index to node features
            if veclen==2:
                child=node.children
                lstar=np.zeros((numleaves,1))
                index0=child[0].pos
                index1=child[1].pos
                lstar[index0]=1
                lstar[index1]=1
                lilmat[i]=np.transpose(lstar)
                haarvec=np.zeros((numleaves,1))
                haarvec[index0]=1/np.sqrt(2)
                haarvec[index1]=-1/np.sqrt(2)
                sparsehaarlike[i]=np.transpose(haarvec)
                print(i)
                i=i+1
            else:
                child=node.children
                if len(node2leaves[child[0]])==1:
                    lstar0=np.zeros((numleaves,1))
                    index0=child[0].pos
                    lstar0[index0]=1
                    index=child[1].loc
                    lstar1=np.transpose(lilmat[index].todense())
                    index1=child[1].pos                
                    lstar1[index1]=lstar1[index1]+len(child[1])*1
                    lilmat[i]=np.transpose(lstar0)+np.transpose(lstar1)
                    L1=np.count_nonzero(lstar1)
                    haarvec=np.zeros((numleaves,1))
                    haarvec[index0]=np.sqrt(L1/(L1+1))
                    haarvec[index1]=-np.sqrt(1/(L1*(L1+1)))
                    sparsehaarlike[i]=np.transpose(haarvec)
                    print(i)
                    i=i+1
                elif len(node2leaves[child[1]])==1:
                    lstar1=np.zeros((numleaves,1))
                    index1=child[1].pos
                    lstar1[index1]=1
                    index=child[0].loc
                    lstar0=np.transpose(lilmat[index].todense())
                    index0=child[0].pos
                    lstar0[index0]=lstar0[index0]+len(child[0])*1
                    lilmat[i]=np.transpose(lstar1)+np.transpose(lstar0)
                    L0=np.count_nonzero(lstar0)
                    haarvec=np.zeros((numleaves,1))
                    haarvec[index0]=np.sqrt(1/(L0*(L0+1)))
                    haarvec[index1]=-np.sqrt(L0/((L0+1)))
                    sparsehaarlike[i]=np.transpose(haarvec)
                    print(i)
                    i=i+1
                else:
                    index0=child[0].loc
                    index1=child[1].loc
                    lstar0=np.transpose(lilmat[index0].todense())
                    lstar1=np.transpose(lilmat[index1].todense())
                    index00=child[0].pos
                    lstar0[index00]=lstar0[index00]+len(child[0])*1
                    index11=child[1].pos
                    lstar1[index11]=lstar1[index11]+len(child[1])*1
                    lilmat[i]=np.transpose(lstar0)+np.transpose(lstar1)
                    L0=np.count_nonzero(lstar0)
                    L1=np.count_nonzero(lstar1)
                    haarvec=np.zeros((numleaves,1))
                    haarvec[index00]=np.sqrt(L1/(L0*(L0+L1)))
                    haarvec[index11]=-np.sqrt(L0/(L1*(L0+L1)))
                    sparsehaarlike[i]=np.transpose(haarvec)
                    print(i)
                    i=i+1
    
    sparsehaarlike[len(allleaves)-1]=np.repeat(1/np.sqrt(len(allleaves)),len(allleaves))
    lilmat=lilmat.tocsr()
    sparsehaarlike=sparsehaarlike.tocsr()
    
    
    return sparsehaarlike


def compute_Haar_dist(mags,weightvec):
    '''

    Parameters
    ----------
    mags : np.ndarray
        Projection of OTU abundance count onto Haar-like basis
    weightvec : np.ndarray
        lamba_v's to use in distance computation

    Returns
    -------
    D : np.ndarray
        Pairwise Haar-like distances
    modmags : scipy.sparse.csr_matrix
        rescaled (by weightvec) projections of OTU abundance onto Haar-like basis

    '''

    modmags=np.transpose(np.asarray(mags.todense())* np.sqrt(weightvec[:, np.newaxis]))
    modmags=scipy.sparse.csr_matrix(modmags)

    N=mags.shape[1]

    #Build Haar-like distance matrix
    D=np.zeros((N,N))
    for i in range(N):
        for j in range(i+1,N):
            distdiff=((modmags[i,:]-modmags[j,:]))
            d=scipy.sparse.csr_matrix.sum(scipy.sparse.csr_matrix.power(distdiff,2))
            D[i,j]=np.sqrt(d)        
    D=D+np.transpose(D)
    return D, modmags
