#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 15:10:08 2023

@author: Evan
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.cm as cm



def magplot(mags,y,indices,xlabel,save,path):
    fig, ax = plt.subplots(len(indices), sharex=True,figsize=(5, 8))
    for i in range(len(indices)):
        ax[i].scatter(y,np.asarray(mags[indices[i],:].todense().squeeze()).T)
        ax[i].set_title('Haar-like Coordinate'+' '+ str(indices[i]))
    plt.xlabel(xlabel)
    if save==True:
        plt.savefig(path, dpi=400,bbox_inches='tight')
    
def boxplot(mags,y,indices,dic,xlabels,save,path):
    fig, ax = plt.subplots(len(indices), sharex=True, figsize=(4, 2.5*len(indices)))
    boxlabels=list(dic.keys())
    inv_map = {v: k for k, v in dic.items()}
    datalabels=[inv_map[i] for i in y]
    for k in range(len(indices)):
        alldata=[]
        for i in range(len(boxlabels)):
            dataindices=[j for j, x in enumerate(datalabels) if x == boxlabels[i]]
            alldata.append(np.asarray(mags[indices[k],dataindices].todense()).squeeze())
        temp=ax[k].boxplot(alldata,
                         vert=True,  # vertical box alignment
                         patch_artist=True,  # fill with color
                         labels=xlabels)  # will be used to label x-ticks
        ax[k].set_title('Haar-like Coordinate'+' '+ str(indices[k]))
        plt.setp(temp['whiskers'], color='black')
        plt.setp(temp['medians'], color='black')
        plt.setp(temp['fliers'], color='black', marker='')
        if len(y)==2:
            colors=cm.tab10(np.array(list(dic.values()))/(2*max(np.array(list(dic.values())))))
        else:
            colors=cm.tab10(np.array(list(dic.values()))/(max(np.array(list(dic.values())))))
        for patch, color in zip(temp['boxes'], colors):
            patch.set_facecolor(color)
    if save==True:
        plt.savefig(path, dpi=400,bbox_inches='tight')

def biplot2d(model,mags,y,labeltype,dic,k,n,save,path):
    Z=np.transpose(model.ReconstructCoord(mags,k))
    #scaler = StandardScaler()
    #scaler.fit(Z)
    #Z=scaler.transform(Z)    
    pca = PCA()
    x_new = pca.fit_transform(np.asarray(Z))
    score=x_new[:,0:2]
    coeff=np.transpose(pca.components_[0:2, :])
    xs = score[:,0]
    ys = score[:,1]
    #scalex = 1.0/(xs.max() - xs.min())
    #scaley = 1.0/(ys.max() - ys.min())
    scalex = 1.0
    scaley = 1.0
    if not dic is None:
        inv_map = {v: k for k, v in dic.items()}
        labels=[inv_map[i] for i in y]
    if labeltype=='classification':
        if len(y)==2:
            colors=cm.tab10(y/(2*max(y)))
        else:
            colors=cm.tab10(y/(max(y)))
        for xs,ys,c,label in zip(xs,ys,colors,labels):
            #plt.scatter(xs*scalex ,ys*scaley ,color=c,label=label)
            plt.scatter(xs*scalex ,ys*scaley ,facecolors="None",edgecolors=c,label=label)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(),prop={'size': 6})

    elif labeltype=='regression': 
        #c = plt.cm.viridis(y/max(y))
        #plt.scatter(xs*scalex ,ys*scaley,  facecolors="None", edgecolors=c)
        plt.scatter(xs ,ys ,c=y, cmap='viridis')
        plt.colorbar()
    for i in range(0,n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],head_width=.03,color = 'r',alpha = 1)
        plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, model.coordinates[i], color = 'black', ha = 'center', va = 'center')    
    #plt.xlim(-.75,.75)
    #plt.ylim(-.75,.75)
    plt.xlabel("PC1"+' '+str(np.around(pca.explained_variance_ratio_[0]*100,2))+'%')
    plt.ylabel("PC2"+' '+str(np.around(pca.explained_variance_ratio_[1]*100,2))+'%')
    plt.title('PCA Biplot')
    plt.grid(False)

    if save==True:
        plt.savefig(path, dpi=400,bbox_inches='tight')
        
def biplot2dnormalized(model,mags,y,labeltype,dic,k,n,save,path):
    Z=np.transpose(model.ReconstructCoord(mags,k))
    scaler = StandardScaler()
    scaler.fit(Z)
    Z=scaler.transform(Z)    
    pca = PCA()
    x_new = pca.fit_transform(Z)
    score=x_new[:,0:2]
    coeff=np.transpose(pca.components_[0:2, :])
    xs = score[:,0]
    ys = score[:,1]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    scalex = 1.0
    scaley = 1.0
    if not dic is None:
        inv_map = {v: k for k, v in dic.items()}
        labels=[inv_map[i] for i in y]
    if labeltype=='classification':
        if len(y)==2:
            colors=cm.tab10(y/(2*max(y)))
        else:
            colors=cm.tab10(y/(max(y)))
        for xs,ys,c,label in zip(xs,ys,colors,labels):
            #plt.scatter(xs*scalex ,ys*scaley ,color=c,label=label)
            plt.scatter(xs*scalex ,ys*scaley ,facecolors="None",edgecolors=c,label=label)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(),prop={'size': 6})

    elif labeltype=='regression': 
        #c = plt.cm.viridis(y/max(y))
        #plt.scatter(xs*scalex ,ys*scaley,  facecolors="None", edgecolors=c)
        plt.scatter(xs*scalex ,ys*scaley ,c=y, cmap='viridis')
        plt.colorbar()
    for i in range(0,n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],head_width=.03,color = 'r',alpha = .7)
        plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, model.coordinates[i], color = 'black', ha = 'center', va = 'center')    
    plt.xlim(-.75,.75)
    plt.ylim(-.75,.75)
    plt.xlabel("PC1"+' '+str(np.around(pca.explained_variance_ratio_[0]*100,2))+'%')
    plt.ylabel("PC2"+' '+str(np.around(pca.explained_variance_ratio_[1]*100,2))+'%')
    plt.title('PCA Biplot')
    plt.grid(False)

    if save==True:
        plt.savefig(path, dpi=400,bbox_inches='tight')
    
    
def biplot3d(model,mags,y,labeltype,dic,k,n,save,path):
    Z=np.transpose(model.ReconstructCoord(mags,k))
    pca = PCA()
    x_new = pca.fit_transform(np.asarray(Z))
    score=x_new[:,0:3]
    coeff=np.transpose(pca.components_[0:3, :])
    xs = score[:,0]
    ys = score[:,1]
    zs = score[:,2]
    scalex = 1.0
    scaley = 1.0
    scalez = 1.0
    fig = plt.figure(figsize=(6, 8))
    ax = fig.add_subplot(projection='3d')
    if not dic is None:
        inv_map = {v: k for k, v in dic.items()}
        labels=[inv_map[i] for i in y]
    if labeltype=='classification':
        if len(y)==2:
            colors=cm.tab10(y/(2*max(y)))
        else:
            colors=cm.tab10(y/max(y))
        for xs,ys,zs,c,label in zip(xs,ys,zs,colors,labels):
            #if label not in ['skin','urine','oral cavity','feces']:
                ax.scatter(xs*scalex ,ys*scaley, zs*scalez ,facecolors="None", edgecolors=c,label=label)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(),prop={'size': 7})

    elif labeltype=='regression': 
        c = plt.cm.viridis(y/max(y))
        p=ax.scatter(xs*scalex ,ys*scaley, zs*scalez,c=y, cmap='viridis')
        plt.colorbar(p, pad = 0.15)
        
    # for i in [5,6]:
    #     ax.quiver(
    #         0, 0, 0, # <-- starting point of vector
    #         1.15*coeff[i,0], 1.15*coeff[i,1], 1.15*coeff[i,2], # <-- directions of vector
    #         color = 'black', alpha = .7, lw = 2
    #     )
    #     ax.text(coeff[i,0]* 1.25, coeff[i,1] * 1.25,coeff[i,2] * 1.25, model.coordinates[i], color = 'black', ha = 'center', va = 'center') 
    for i in range(n):
        ax.quiver(
            0, 0, 0, # <-- starting point of vector
            1.15*coeff[i,0], 1.15*coeff[i,1], 1.15*coeff[i,2], # <-- directions of vector
            color = 'black', alpha = .7, lw = 2
        )
        ax.text(coeff[i,0]* 1.25, coeff[i,1] * 1.25,coeff[i,2] * 1.25, model.coordinates[i], color = 'black', ha = 'center', va = 'center') 
    
    plt.xlabel("PC1"+' '+str(np.around(pca.explained_variance_ratio_[0]*100,2))+'%')
    plt.ylabel("PC2"+' '+str(np.around(pca.explained_variance_ratio_[1]*100,2))+'%')
    ax.set_zlabel("PC3"+' '+str(np.around(pca.explained_variance_ratio_[2]*100,2))+'%')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 6, box.height])
    
    # for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
    #     axis.set_ticklabels([])
    #     axis._axinfo['axisline']['linewidth'] = 1
    #     axis._axinfo['axisline']['color'] = (0, 0, 0)
    #     axis._axinfo['grid']['linewidth'] = 0.5
    #     axis._axinfo['grid']['linestyle'] = "-"
    #     axis._axinfo['grid']['color'] = (0, 0, 0)
    #     axis._axinfo['tick']['inward_factor'] = 0.0
    #     axis._axinfo['tick']['outward_factor'] = 0.0
    
    #plt.legend(['Skin', 'EAC','Nostril','Hair','Urine','Oral Cavity','Feces'],loc='center left', bbox_to_anchor=(-.5, 0.5))
    plt.title('PCA Biplot')
    ax.grid(False)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.set_facecolor('white')
    ax.dist=12
    fig.tight_layout(pad=2)
    #plt.savefig('/Users/Evan/Desktop/AdaptHaarFigs/Costello/zoom', dpi=400,bbox_inches='tight')

    if save==True:
        plt.savefig(path, dpi=400,bbox_inches='tight')

    
def biplot3dnormalized(model,mags,y,labeltype,dic,k,n,save,path):
    Z=np.transpose(model.ReconstructCoord(mags,k))
    scaler = StandardScaler()
    scaler.fit(np.asarray(Z))
    Z=scaler.transform(np.asarray(Z))  
    pca = PCA()
    x_new = pca.fit_transform(Z)
    score=x_new[:,0:3]
    coeff=np.transpose(pca.components_[0:3, :])
    xs = score[:,0]
    ys = score[:,1]
    zs = score[:,2]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    scalez = 1.0/(zs.max() - ys.min())
    fig = plt.figure(figsize=(6, 8))
    ax = fig.add_subplot(projection='3d')
    if not dic is None:
        inv_map = {v: k for k, v in dic.items()}
        labels=[inv_map[i] for i in y]
    if labeltype=='classification':
        if len(y)==2:
            colors=cm.tab10(y/(2*max(y)))
        else:
            colors=cm.tab10(y/max(y))
        for xs,ys,zs,c,label in zip(xs,ys,zs,colors,labels):
            #if label not in ['skin','urine','oral cavity','feces']:
                ax.scatter(xs*scalex ,ys*scaley, zs*scalez ,facecolors="None", edgecolors=c,label=label)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(),prop={'size': 7})

    elif labeltype=='regression': 
        c = plt.cm.viridis(y/max(y))
        p=ax.scatter(xs*scalex ,ys*scaley, zs*scalez,c=y, cmap='viridis')
        plt.colorbar(p, pad = 0.15)
    
  
    for i in range(n):
        ax.quiver(
            0, 0, 0, # <-- starting point of vector
            1.15*coeff[i,0], 1.15*coeff[i,1], 1.15*coeff[i,2], # <-- directions of vector
            color = 'black', alpha = .7, lw = 2
        )
        ax.text(coeff[i,0]* 1.25, coeff[i,1] * 1.25,coeff[i,2] * 1.25, model.coordinates[i], color = 'black', ha = 'center', va = 'center') 
    plt.xlabel("PC1"+' '+str(np.around(pca.explained_variance_ratio_[0]*100,2))+'%')
    plt.ylabel("PC2"+' '+str(np.around(pca.explained_variance_ratio_[1]*100,2))+'%')
    ax.set_zlabel("PC3"+' '+str(np.around(pca.explained_variance_ratio_[2]*100,2))+'%')

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 6, box.height])
    plt.title('PCA Biplot Normalized')
    ax.grid(False)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.set_facecolor('white')
    ax.dist=12
    fig.tight_layout(pad=2)    
    if save==True:
        plt.savefig(path, dpi=400,bbox_inches='tight')

    
    
    
def PCoA(dist,n):
    A2=np.square(dist)
    J=np.eye(len(dist))-1/(len(dist))*np.ones(len(dist))
    B=-1/2*J@A2@J
    [D,V]=np.linalg.eigh(B)
    D=np.real(D)
    D=np.flip(D)
    V=np.flip(V,axis=1)
    V=V[:,0:n]
    Dsqrt=np.sqrt(D[0:n])
    X=V@np.diag(Dsqrt)
    
    return X, [D[0]/sum(D),D[1]/sum(D),D[2]/sum(D)]



def Unifracplot2d(D,y,dic,tasktype,title,save,path):
    X,perc=PCoA(D,2)
    xs = X[:,0]
    ys = X[:,1]
    if not dic is None:
        inv_map = {v: k for k, v in dic.items()}
        labels=[inv_map[i] for i in y]
    if tasktype=='classification':
        if len(y)==2:
            colors=cm.tab10(y/(2*max(y)))
        else:
            colors=cm.tab10(y/(max(y)))
        for xs,ys,c,label in zip(xs,ys,colors,labels):
            #plt.scatter(xs*scalex ,ys*scaley ,color=c,label=label)
            plt.scatter(xs ,ys ,facecolors="None",edgecolors=c,label=label)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(),prop={'size': 6})

    elif tasktype=='regression': 
        #c = plt.cm.viridis(y/max(y))
        #plt.scatter(xs*scalex ,ys*scaley,  facecolors="None", edgecolors=c)
        plt.scatter(xs ,ys ,c=y, cmap='viridis')
        plt.colorbar()
    #plt.xlim(-.6,.75)
    #plt.ylim(-.75,.75)
    plt.xlabel("PC1"+' '+str(np.around(perc[0]*100,2))+'%')
    plt.ylabel("PC2"+' '+str(np.around(perc[1]*100,2))+'%')
    plt.title('PCA Biplot')
    plt.grid(False)
    if save==True:
        plt.savefig(path, dpi=400,bbox_inches='tight')


def Unifracplot3d(D,y,dic,tasktype,title,save,path):
    X,perc=PCoA(D,3)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection='3d')
    xs = X[:,0]
    ys = X[:,1]
    zs = X[:,2]
    if not dic is None:
        inv_map = {v: k for k, v in dic.items()}
        labels=[inv_map[i] for i in y]
    if tasktype=='classification':
        if len(y)==2:
            colors=cm.tab10(y/(2*max(y)))
        else:
            colors=cm.tab10(y/max(y))
        for xs,ys,zs,c,label in zip(xs,ys,zs,colors,labels):
            ax.scatter(xs ,ys, zs ,facecolors="None", edgecolors=c,label=label)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(),prop={'size': 7})

    elif tasktype=='regression': 
        c = plt.cm.viridis(y/max(y))
        p=ax.scatter(xs ,ys, zs,c=y, cmap='viridis')
        plt.colorbar(p, pad = 0.15)
    
    
    plt.xlabel("PC1"+' '+str(np.around(perc[0]*100,2))+'%',fontsize=8)
    plt.ylabel("PC2"+' '+str(np.around(perc[1]*100,2))+'%',fontsize=8)
    ax.set_zlabel("PC3"+' '+str(np.around(perc[2]*100,2))+'%',fontsize=8)
    #PCM=ax.get_children()[2]
    #plt.colorbar(PCM, ax=ax) 
    #ax.colorbar()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 6, box.height])
    #plt.legend(['Skin', 'EAC','Nostril','Hair','Urine','Oral Cavity','Feces'],loc='center left', bbox_to_anchor=(-.5, 0.5))
    plt.title(title)
    ax.grid(False)
    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.set_facecolor('white')
    fig.tight_layout(pad=2)    
    ax.dist=14
    if save==True:
        plt.savefig(path, dpi=400,bbox_inches='tight')

