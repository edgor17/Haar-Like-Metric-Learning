#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Evan




NOTE: CodaCore for python is incompatible with the packages necessary
for plotting in AdaptiveHaarLike. To run this script we need a seperate 
environment without the plotting packages.
"""

import scipy 
from ete3 import Tree
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import metrics
from codacore.model import CodaCore
import numpy as np
import sklearn

from AdaptiveHaarLike.model import AdaptiveHaarLike

def ModelComp(X,Xnorm,Y,mags):
    count=0
    CODAerror=[]
    CODAAUC=[]
    CODAsparsity=[]
    RFerror=[]
    RFAUC=[]
    HAARerror=[]
    HAARAUC=[]
    beststrack=[]
    indices=[]
    kf = RepeatedStratifiedKFold(n_splits=5,n_repeats=5)
    for i, (train_index, test_index) in enumerate(kf.split(X, Y)):
        
        #CODA MODEL
        Xtemp=X.values
        Ytemp=Y.values.astype(int)
        Xtemp=Xtemp+np.ones((len(Xtemp),np.shape(Xtemp)[1]))
        try:
            model = CodaCore(objective='binary_classification', type='balance',regularization=0)
            model.fit(Xtemp[train_index],Ytemp[train_index])
            YpredCODA=model.predict(Xtemp[test_index],return_logits=False)
            CODAsparsity.append(len(model.ensemble))
            
            CODAAUC.append(metrics.roc_auc_score(Ytemp[test_index],YpredCODA))
            YpredCODA[YpredCODA<.5]=0
            YpredCODA[YpredCODA>=.5]=1
            CODAerror.append(metrics.accuracy_score(YpredCODA,Ytemp[test_index]))
        except:
            CODAAUC.append(np.nan)
            CODAerror.append(np.nan)
        
        #RF MODEL
        clf=RandomForestClassifier(n_estimators=500,bootstrap=True,min_samples_leaf=1,max_features=None)
        clf.fit(Xnorm.iloc[train_index],Y.iloc[train_index])
        YpredRF=clf.predict(Xnorm.iloc[test_index])
        
        RFerror.append(metrics.accuracy_score(YpredRF,Y.iloc[test_index]))
        RFAUC.append(metrics.roc_auc_score(Y.iloc[test_index],clf.predict_proba(X.iloc[test_index])[:,1]))
        
        
        #HAAR-like RF
        model = AdaptiveHaarLike(labeltype='classification')
        model.fit(clf,Xnorm,Y,10,mags)
        
        
        
        bests=0
        avgacc=0    
        for s in range(1,11):
            kf2 = RepeatedStratifiedKFold(n_splits=5,n_repeats=5)
            acc=0
            for i2, (train_index_inner, test_index_inner) in enumerate(kf2.split(X.iloc[train_index], Y.iloc[train_index])):
                Ypred=model.Predict(mags,train_index_inner,test_index_inner,Y,s,int(np.floor(np.sqrt(len(train_index_inner)))))       
                acc=acc+metrics.accuracy_score(np.sign(np.array(Ypred)),(Y.iloc[test_index_inner].values)*2-1)
            newavgacc=acc/25
            if newavgacc>avgacc:
                avgacc=newavgacc
                bests=s
                
                
        Ypred=model.Predict(mags,train_index,test_index,Y,bests,int(np.floor(np.sqrt(len(train_index)))))        
        HAARerror.append(metrics.accuracy_score(np.sign(np.array(Ypred)),(Y.iloc[test_index].values)*2-1))
        HAARAUC.append(metrics.roc_auc_score(Y.iloc[test_index],(np.array(Ypred)+1)/2))
        indices.append(model.coordinates[0])
        
        beststrack.append(bests)


        print(HAARerror[-1],CODAerror[-1],RFerror[-1])
        print(count)
    return CODAerror,CODAAUC,np.mean(CODAsparsity),RFerror,RFAUC,HAARerror,HAARAUC,np.mean(beststrack),scipy.stats.mode(indices)[0][0]


folder='/Users/Evan/Desktop/MLRepo-master/datasets/'
tree = Tree("/Users/Evan/Sparsify_Ultrametric/raw_data/97_otus_unannotated.tree",format=1)
haarlike=scipy.sparse.load_npz('/Users/Evan/Sparsify_Ultrametric/precomputed/97haarlike.npz')
paths=pd.read_csv('/Users/Evan/Desktop/AdaptHaar/pathstorepo.txt', sep='\t')
data=[]

for index, row in paths.iterrows():
    featuretable=pd.read_csv(folder+row['feature_table'], sep='\t')            
    metadata=pd.read_csv(folder+row['metadata'], sep='\t',dtype={'#SampleID': str})            
    [X,Y,mags,dic]=PreProcess(featuretable,metadata,'Var','classification',tree,haarlike)
    Xnorm=X
    Xnorm=Xnorm.div(Xnorm.sum(axis=1), axis=0)
    CODAerror,CODAAUC,CODAsparsity,RFerror,RFAUC,HAARerror,HAARAUC,HAARsparsity,bestsindex=ModelComp(X,Xnorm,Y.astype(int)-1,mags)
    data.append([CODAerror,CODAAUC,CODAsparsity,RFerror,RFAUC,HAARerror,HAARAUC,HAARsparsity,bestsindex,row['metadata']])

results=pd.DataFrame(data,columns=['CODA Error','CODA AUC','CODA sparsity','RF Error','RF AUC','Haar Error','Haar AUC','Haar Sparsity','Index','task'])