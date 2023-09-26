#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 11:24:55 2023

@author: Evan
"""
import scipy as scipy
import numpy as np
from sklearn.manifold import MDS



class AdaptiveHaarLike:
    def __init__(
            self,
            labeltype,
        ):
        self.coefs=[]
        self.importances=[]
        self.coordinates=[]
        self.labeltype=labeltype

    def fit(self, clf, X, Y, s, mags):
        '''
        Fits the Adaptive Haar-like metric to a dataset

        Parameters
        ----------
        clf : ensemble._forest.RandomForestClassifier
        X : np.ndarray
            Normalized OTU abundance table
        s : int
            Number of Haar-like Coordinates to use
        mags : scipy.sparse.csr_matrix
            Haar-like mags
        nsamples : int
            Number of samples to use in fit
        '''

        self.s=s
        self.rfaffinity=self.proximityMatrix(clf, X)
        self.clf=clf
        [self.signal, self.dictionary, self.samples, self.rfgram]=self.ConvertLeastSquares(self.rfaffinity,mags,len(X))
        [self.coordinates, self.coefs, self.importances, self.R]=self.MatchingPursuit(scipy.sparse.csc_matrix(self.signal), self.dictionary)

        return self

    def spouter(self,A,B):
        '''
        Quickly compute sparse outer product. This code was obtained from https://stackoverflow.com/questions/57099722/row-wise-outer-product-on-sparse-matrices
        '''
        import itertools
        N,L = A.shape
        N,K = B.shape
        drows = zip(*(np.split(x.data,x.indptr[1:-1]) for x in (A,B)))
        data = [np.outer(a,b).ravel() for a,b in drows]
        irows = zip(*(np.split(x.indices,x.indptr[1:-1]) for x in (A,B)))
        indices = [np.ravel_multi_index(np.ix_(a,b),(L,K)).ravel() for a,b in irows]
        indptr = np.fromiter(itertools.chain((0,),map(len,indices)),int).cumsum()
        return scipy.sparse.csr_matrix((np.concatenate(data),np.concatenate(indices),indptr),(N,L*K))

    def proximityMatrix(self,model, X):     
        '''
        Generate random forest affinity matrix
        '''
        
        terminals = model.apply(X)
        nTrees = terminals.shape[1]

        a = terminals[:,0]
        proxMat = 1*np.equal.outer(a, a)

        for i in range(1, nTrees):
            a = terminals[:,i]
            proxMat += 1*np.equal.outer(a, a)

        proxMat = proxMat / nTrees

        return proxMat 
    
    
    def ConvertLeastSquares(self,affinity,mags,samplesize):
        '''
        Converts the Frobenius optimization problem into a least squares problem. 
        This step may be expensive for very large sample sizes, so there is an option
        to subsample using the nystrom method.
        '''
        [d,n]=mags.get_shape()

        #randomly subsample if necessary
        arr=np.arange(n)
        np.random.shuffle(arr)
        subsamples=arr[0:samplesize]
        subsamples=np.sort(subsamples)
        
        #compute RF gram matrix
        affinity=affinity[np.ix_(subsamples,subsamples)]
        embedding = MDS(n_components=50,dissimilarity='precomputed')
        distances=1-affinity
        X_transformed = embedding.fit_transform(distances)
        sparsegram=scipy.sparse.csr_matrix(X_transformed@np.transpose(X_transformed))

        #Vectorize
        signal=scipy.sparse.csr_matrix.reshape(sparsegram,((samplesize**2,1)),order='F')
        mags=mags[:,subsamples]
        print('Building Dictionary')
        C=self.spouter(mags,mags)
        A=scipy.sparse.csc_matrix(scipy.sparse.csr_matrix.transpose(C))
        return signal, A, subsamples,sparsegram




    def MatchingPursuit(self, signal, dictionary):
        from sklearn.preprocessing import normalize
        dictionarynorm=normalize(dictionary,norm='l2',axis=0)
        coefs=[]
        indices=[]
        importances=[]
        R=signal
        for i in range(self.s):
            print(i)
            innerprod=scipy.sparse.csc_matrix.transpose(dictionarynorm)@R
            index=np.argmax((innerprod))
            indices.append(index)
            maxproj=innerprod[index].todense().item()
            importances.append(maxproj)
            coefs.append(maxproj/scipy.sparse.linalg.norm(dictionary[:,index]))
            R=R-maxproj*dictionarynorm[:,index]
        return indices, coefs, importances, R

    #Build the Haar gram matrix from the learned weights and coordinates
    def Reconstruct(self,mags,s):
        [d,n]=mags.get_shape()
        outer=np.zeros((n,n))
        print("Reconstructing")
        for i in range(s):
            print(i)
            temp=mags[self.coordinates[i],:].todense()
            out=np.outer(temp,temp)
            outer=outer+self.coefs[i]*out
        return outer
    


    #Directly compute coordinates of the learned embedding from the learned weights 
    #and coordinates
    def ReconstructCoord(self,mags,s):
        [d,n]=mags.get_shape()
        print("Reconstructing")
        Coord=np.sqrt(self.coefs[0])*mags[self.coordinates[0],:].todense()
        for i in range(1,s):
            Coord=np.vstack((Coord,np.sqrt(self.coefs[i])*mags[self.coordinates[i],:].todense()))
        return Coord
        
    def Predict(self,mags,train_index,test_index,Y,s,k):
        coord=self.ReconstructCoord(mags,s)
        labels=np.copy(Y.values)
        labels=2*labels-1
        D=scipy.spatial.distance_matrix(coord.T,coord.T)
        D[D>1]=1
        classification=[]
        for i in test_index:
            PKNN=np.argsort(D[i,train_index])
            KNN=train_index[PKNN][0:k]
            vote=0
            trackweight=0
            for index in KNN:
                weight=1-D[i,index]
                vote=vote+weight*labels[index]
                trackweight=trackweight+weight
            if not trackweight==0:
                vote=vote/trackweight
            classification.append((vote))
        return classification