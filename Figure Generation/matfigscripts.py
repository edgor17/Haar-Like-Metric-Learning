#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 16:29:48 2023

@author: Evan
"""
import pandas as pd
import scipy as scipy
import scipy.sparse
import numpy as np
from ete3 import Tree
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

folder='/Users/Evan/Desktop/AdaptHaarFigs/Mat'

featuretable=pd.read_csv("/Users/Evan/Desktop/soildepth/otu_table.txt", sep='\t')
metadata=pd.read_csv("/Users/Evan/Desktop/soildepth/metadata.txt", sep='\t')
label='matdepthcryosectionmm'
labeltype='regression'
tree = Tree("/Users/Evan/Sparsify_Ultrametric/raw_data/97_otus_unannotated.tree",format=1)
haarlike=scipy.sparse.load_npz('/Users/Evan/Sparsify_Ultrametric/precomputed/97haarlike.npz')
pseudodiag=scipy.sparse.load_npz('/Users/Evan/Sparsify_Ultrametric/precomputed/97pseudodiag.npz')
lambdav=scipy.sparse.csr_matrix.diagonal(pseudodiag)

[X,Y,mags,dic]=PreProcess(featuretable,metadata,label,labeltype,tree,haarlike)
model = AdaptiveHaarLike(labeltype)
clf=RandomForestRegressor(n_estimators=500,bootstrap=True,min_samples_leaf=27)
clf.fit(X,Y)
model.fit(clf,X,Y,50,mags)

plt.imshow(model.rfgram.todense(),vmin=0,vmax=.3,cmap='binary')
plt.savefig(folder+'/rfgram')
plt.imshow(model.Reconstruct(mags,2),vmin=0,vmax=.3,cmap='binary')
plt.savefig(folder+'/reconstruct2')
plt.imshow(model.Reconstruct(mags,50),vmin=0,vmax=.3,cmap='binary')
plt.savefig(folder+'/reconstruct50')

ys=model.importances

def monoExp(x, m, t, b):
    return m * np.exp(-t * x) + b
# perform the fit
xs=np.linspace(1,len(ys),len(ys))
p0 = (2000, .1, 50) # start with values near those we expect
params, cv = scipy.optimize.curve_fit(monoExp, xs, ys, p0)
m, t, b = params
sampleRate = 20_000 # Hz
tauSec = (1 / t) / sampleRate

# determine quality of the fit
squaredDiffs = np.square(ys - monoExp(xs, m, t, b))
squaredDiffsFromMean = np.square(ys - np.mean(ys))
rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)
print(f"R² = {rSquared}")
f, ax = plt.subplots()
# plot the results
ax.plot(xs, ys, '.', label="Haar-like Importances")
ax.plot(xs, monoExp(xs, m, t, b), '--', label="Exponential Fit")
ax.legend()
plt.title("Haar-like Coordinate Importance Decay")
plt.savefig(folder+'/importances')
# inspect the parameters
print(f"Y = {m} * e^(-{t} * x) + {b}")
print(f"Tau = {tauSec * 1e6} µs")

truedist=scipy.spatial.distance_matrix(Y.values[:,np.newaxis],Y.values[:,np.newaxis])

Dunifrac=np.load(folder+'/unifracsort.npy')
dcor(truedist,Dunifrac)
Unifracplot2d(Dunifrac,dic=None,y=Y.values,tasktype='regression',title='UniFrac',save=True,path=folder+'/unifracplot')
Dwunifrac=np.load(folder+'/wunifracsort.npy')
dcor(truedist,Dwunifrac)
Unifracplot2d(Dwunifrac,dic=None,y=Y.values,tasktype='regression',title='Weighted UniFrac',save=True,path=folder+'/wunifracplot')
[Dhaar,modmags]=compute_Haar_dist(mags,lambdav)
dcor(truedist,Dhaar)
Unifracplot2d(Dhaar,dic=None,y=Y.values,tasktype='regression',title='Haar-like Distance',save=True,path=folder+'/haardistplot')


dcor(truedist,scipy.spatial.distance_matrix(model.ReconstructCoord(mags,2).T,model.ReconstructCoord(mags,2).T))


biplot2d(model,mags,Y.values.astype(float),'regression',dic,k=2,n=2,save=True,path=folder+'/biplot')
magplot(mags,Y.values,model.coordinates[0:2],'Distance From Wellhead',True,folder+'/magplot')
