#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 14:54:20 2023

@author: Evan
"""
import pandas as pd
import scipy as scipy
import scipy.sparse
import numpy as np
from ete3 import Tree
from skbio import DistanceMatrix
from skbio.stats.distance import permanova
import matplotlib.pyplot as plt

from WHAT import PreProcess
from WHAT import AdaptiveHaarLike

folder='/Users/Evan/Desktop/AdaptHaarFigs/Costello'

featuretable=pd.read_csv("/Users/Evan/Desktop/costello/costello.txt", sep='\t')
metadata=pd.read_csv("/Users/Evan/Desktop/costello/metadata.txt", sep='\t')
label='host_body_habitat'
labeltype='classification'
tree = Tree("/Users/Evan/Sparsify_Ultrametric/raw_data/97_otus_unannotated.tree",format=1)
haarlike=scipy.sparse.load_npz('/Users/Evan/Sparsify_Ultrametric/precomputed/97haarlike.npz')
pseudodiag=scipy.sparse.load_npz('/Users/Evan/Sparsify_Ultrametric/precomputed/97pseudodiag.npz')
lambdav=scipy.sparse.csr_matrix.diagonal(pseudodiag)

[X,Y,mags,dic]=PreProcess(featuretable,metadata,label,labeltype,tree,haarlike)
X=X.div(X.sum(axis=1), axis=0)
model = AdaptiveHaarLike(labeltype)
clf=RandomForestClassifier(n_estimators=500,bootstrap=True,min_samples_leaf=1)
clf.fit(X,Y)
model.fit(clf,X,Y,10,mags)


plt.imshow(model.rfgram.todense(),vmin=0,vmax=.2,cmap='binary')
plt.savefig(folder+'/rfgram')
plt.imshow(model.Reconstruct(mags,7),vmin=0,vmax=.2,cmap='binary')
plt.savefig(folder+'/reconstruct7')
plt.imshow(model.Reconstruct(mags,50),vmin=0,vmax=.2,cmap='binary')
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


Dunifrac=np.load(folder+'/unifracsort.npy')
permanova(DistanceMatrix(Dunifrac),Y.values)
Unifracplot3d(Dunifrac,y=Y.values,dic=dic,tasktype='classification',title='UniFrac',save=True,path=folder+'/unifracplot')

Dwunifrac=np.load(folder+'/wunifracsort.npy')
permanova(DistanceMatrix(Dwunifrac),Y.values)
Unifracplot3d(Dwunifrac,y=Y.values,dic=dic,tasktype='classification',title='Weighted UniFrac',save=True,path=folder+'/wunifracplot')

[Dhaar,modmags]=compute_Haar_dist(mags,lambdav)
permanova(DistanceMatrix(Dhaar),Y.values)
Unifracplot3d(Dhaar,y=Y.values,dic=dic,tasktype='classification',title='Haar-like Distance',save=True,path=folder+'/haardistplot')

Dadapthaar=scipy.spatial.distance_matrix(model.ReconstructCoord(mags,7).T,model.ReconstructCoord(mags,7).T)
permanova(DistanceMatrix(Dadapthaar),Y.values)


biplot3d(model,mags,Y.values.astype(float),'classification',dic,k=7,n=7,save=True,path=folder+'/biplot')


biplot3dnormalized(model,mags,Y.values.astype(float),'classification',dic,k=7,n=7,save=True,path=folder+'/biplotnormalized')

boxplot(mags,Y.values,model.coordinates[0:7],dic,dic.keys(),True,'/Users/Evan/Desktop/AdaptHaarFigs/Costello/boxplot')

