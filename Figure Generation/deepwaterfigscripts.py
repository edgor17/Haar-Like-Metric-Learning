#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 10:18:43 2023

@author: Evan
"""
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
import matplotlib.pyplot as plt

folder='/Users/Evan/Desktop/AdaptHaarFigs/Deepwater'

featuretable=pd.read_csv("/Users/Evan/Desktop/deepwaterfinal/otus.txt", sep='\t')
metadata=pd.read_csv("/Users/Evan/Desktop/deepwaterfinal/metadata.txt", sep='\t')
label='distance'
labeltype='regression'
tree = Tree("/Users/Evan/Sparsify_Ultrametric/raw_data/97_otus_unannotated.tree",format=1)
haarlike=scipy.sparse.load_npz('/Users/Evan/Sparsify_Ultrametric/precomputed/97haarlike.npz')
pseudodiag=scipy.sparse.load_npz('/Users/Evan/Sparsify_Ultrametric/precomputed/97pseudodiag.npz')
lambdav=scipy.sparse.csr_matrix.diagonal(pseudodiag)

[X,Y,mags,dic]=PreProcess(featuretable,metadata,label,labeltype,tree,haarlike)
X=X.div(X.sum(axis=1), axis=0)
model = AdaptiveHaarLike(labeltype)
clf=RandomForestRegressor(n_estimators=500,bootstrap=True,min_samples_leaf=11)
clf.fit(X,Y)
model.fit(clf,X,Y,10,mags)

plt.imshow(model.rfgram.todense(),vmin=0,vmax=.3,cmap='binary')
plt.savefig(folder+'/rfgram')
plt.imshow(model.Reconstruct(mags,4),vmin=0,vmax=.3,cmap='binary')
plt.savefig(folder+'/reconstruct4')
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
Unifracplot2d(Dunifrac,dic=None,y=np.log(Y.values),tasktype='regression',title='UniFrac',save=True,path=folder+'/unifracplot')
Dwunifrac=np.load(folder+'/wunifracsort.npy')
dcor(truedist,Dwunifrac)
Unifracplot2d(Dwunifrac,dic=None,y=np.log(Y.values),tasktype='regression',title='Weighted UniFrac',save=True,path=folder+'/wunifracplot')
[Dhaar,modmags]=compute_Haar_dist(mags,lambdav)
dcor(truedist,Dhaar)
Unifracplot2d(Dhaar,dic=None,y=np.log(Y.values),tasktype='regression',title='Haar-like Distance',save=True,path=folder+'/haardistplot')


dcor(truedist,scipy.spatial.distance_matrix(model.ReconstructCoord(mags,4).T,model.ReconstructCoord(mags,4).T))


biplot2d(model,mags,np.log(Y.values.astype(float)),'regression',dic,k=4,n=4,save=True,path=folder+'/biplot')
magplot(mags,np.log(Y.values),model.coordinates[0:4],'Log Distance From Wellhead',True,folder+'/magplot')


plt.scatter(np.linspace(1,len(Y),len(Y)),np.log(Y.values))
plt.xlabel('Sample Index')
plt.ylabel('Log of Distance from Wellhead')
plt.savefig('/Users/Evan/Desktop/AdaptHaarFigs/Deepwater/logdists')