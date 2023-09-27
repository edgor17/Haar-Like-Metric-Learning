#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 09:51:06 2023

@author: Evan
"""
import pandas as pd
import scipy as scipy
import scipy.sparse
import numpy as np
from ete3 import Tree
import matplotlib.pyplot as plt
from skbio import DistanceMatrix
from skbio.stats.distance import permanova
from sklearn.ensemble import RandomForestClassifier


from AdaptiveHaarLike import utils
from AdaptiveHaarLike.model import AdaptiveHaarLike
from AdaptiveHaarLike import plotters

folder='Haar-Like-Metric-Learning/Raw_data/ASD'

featuretable=pd.read_csv("Haar-Like-Metric-Learning/Raw_data/ASD/otus.txt", sep='\t')
metadata=pd.read_csv("Haar-Like-Metric-Learning/Raw_data/ASD/metadata.txt", sep='\t')
label='diagnosis'
labeltype='classification'
tree = Tree("Haar-Like-Metric-Learning/raw_data/97_otus_unannotated.tree",format=1)
haarlike=scipy.sparse.load_npz('Haar-Like-Metric-Learning/precomputed/97haarlike.npz')
pseudodiag=scipy.sparse.load_npz('Haar-Like-Metric-Learning/precomputed/97pseudodiag.npz')
lambdav=scipy.sparse.csr_matrix.diagonal(pseudodiag)

[X,Y,mags,dic]=utils.PreProcess(featuretable,metadata,label,labeltype,tree,haarlike)
X=X.div(X.sum(axis=1), axis=0)
model = AdaptiveHaarLike(labeltype)
clf=RandomForestClassifier(n_estimators=500,bootstrap=True,min_samples_leaf=1)
clf.fit(X,Y)
model.fit(clf,X,Y,50,mags)


plt.imshow(model.rfgram.todense(),vmin=0,vmax=.3,cmap='binary')
#plt.savefig(folder+'/rfgram')
plt.imshow(model.Reconstruct(mags,3),vmin=0,vmax=.3,cmap='binary')
#plt.savefig(folder+'/reconstruct3')
plt.imshow(model.Reconstruct(mags,50),vmin=0,vmax=.3,cmap='binary')
#plt.savefig(folder+'/reconstruct50')

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
#plt.savefig(folder+'/importances')
# inspect the parameters
print(f"Y = {m} * e^(-{t} * x) + {b}")
print(f"Tau = {tauSec * 1e6} µs")



Dunifrac=np.load(folder+'/unifracsort.npy')
permanova(DistanceMatrix(Dunifrac),Y.values)
plotters.Unifracplot3d(Dunifrac,dic=dic,y=Y.values,tasktype='classification',title='UniFrac',save=False,path=False)
Dwunifrac=np.load(folder+'/wunifracsort.npy')
permanova(DistanceMatrix(Dwunifrac),Y.values)
plotters.Unifracplot3d(Dwunifra,dic=dic,y=Y.values,tasktype='classification',title='Weighted UniFrac',save=False,path=False)
[Dhaar,modmags]=utils.compute_Haar_dist(mags,lambdav)
permanova(DistanceMatrix(Dhaar),Y.values)
plotters.Unifracplot3d(Dhaar,dic=dic,y=Y.values,tasktype='classification',title='Haar-like Distance',save=False,path=False)

Dadapthaar=scipy.spatial.distance_matrix(model.ReconstructCoord(mags,3).T,model.ReconstructCoord(mags,3).T)
permanova(DistanceMatrix(Dadapthaar),Y.values)

plotters.biplot3d(model,mags,Y.values.astype(float),'classification',dic,k=3,n=3,save=False,path=False)
plotters.biplot3dnormalized(model,mags,Y.values.astype(float),'classification',dic,k=3,n=3,save=False,path=False)

plotters.boxplot(mags,Y.values,model.coordinates[0:3],dic,dic.keys(),False,False)


