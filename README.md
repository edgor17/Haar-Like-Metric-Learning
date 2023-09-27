# Adaptive Haar-like Metric Learning

The Adaptive Haar-like metric learns a data dependent set of Haar-coordinates and associated weights to best represent a given dataset. 

## Installation
```
git clone https://github.com/edgor17/Haar-like-Metric-Learning
cd Haar-Like-Metric-Learning
pip install .
```

# Tutorial for a Classification Type Dataset

We first demonstrate our metric on a dataset from "Bacterial community variation in human body habitats across space and time." This data is publically available on QIITA (ID 449). For this example we will show how the Haar-like metric can distinguish between different body habitats that the samples were taken from. 

## Preprocessing
First we need to generate a Haar-like basis for the phylogeny associated with this dataset. In this case, Greengenes 97 was the reference phylogeny. We use ete toolkit to process .nwk trees. Note that this step only needs to be done once for any phylogenetic tree. For convenience we include a precomputed Haar-like basis for Greengenes 97.

```
import scipy
haarlike=scipy.sparse.load_npz('Haar-Like-Metric-Learning/precomputed/97haarlike.npz')
```

Next we map the given feature table onto the reference phylogeny and sort by the label of interest. Here we use host_body_habitat. X is a dataframe holding the count data mapped to the leaves of Greengenes 97. We will need to normalize this before training our model. Y contains the sample labels. mags is the result of applying the Haar-like transformation to the feature table. dic is a dictionary containing an integer mapping of the unique metadata labels of interest, this will be useful for plotting. 

```
import pandas as pd
from ete3 import Tree
from AdaptiveHaarLike import utils
featuretable=pd.read_csv("Haar-Like-Metric-Learning/CostelloBodySites/costello.txt", sep='\t')
metadata=pd.read_csv("Haar-Like-Metric-Learning/CostelloBodySites/metadata.txt", sep='\t')
label='host_body_habitat'
labeltype='classification'
tree = Tree("Haar-Like-Metric-Learning/raw_data/97_otus_unannotated.tree",format=1)
haarlike=scipy.sparse.load_npz('Haar-Like-Metric-Learning/precomputed/97haarlike.npz')
[X,Y,mags,dic]=utils.PreProcess(featuretable,metadata,label,labeltype,tree,haarlike)
```

## Training 
Now we are ready to train our metric. First we train a random forest classifier with 500 trees and min_samples_leaf set to 1 (this is recommended for classification tasks). Next we fit our model to the random forest with a desired sparsity s=10.

```
from sklearn.ensemble import RandomForestClassifier
from AdaptiveHaarLike.model import AdaptiveHaarLike
X=X.div(X.sum(axis=1), axis=0)
clf=RandomForestClassifier(n_estimators=500,bootstrap=True,min_samples_leaf=1)
clf.fit(X,Y)
model = AdaptiveHaarLike(labeltype)
model.fit(clf,X,Y,10,mags)
```

At this point we can compare the Random Forest Gram matrix to our learned Gram matrix

```
import matplotlib.pyplot as plt
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 5))
axes[0].imshow(model.rfgram.todense(),vmin=0,vmax=.3,cmap='binary')
axes[1].imshow(model.Reconstruct(mags,10),vmin=0,vmax=.3,cmap='binary')
axes[0].xaxis.set_visible(False)
axes[1].xaxis.set_visible(False)
fig.tight_layout()
print(list(dic.keys()))
```

![rfgrambodysites](https://github.com/edgor17/Haar-Like-Metric-Learning/assets/87628022/460c0b89-4e08-4d9f-837f-8121fdd1d2c6)

['skin', 'external auditory canal', 'feces', 'hair', 'oral cavity', 'nostril', 'urine']

## Results
Having learned a Haar-like Gram matrix, we are now ready to plot the embedding. Here we set n=7 to plot the 7 dominant Haar-like coordinate loadings. 

```
from AdaptiveHaarLike import plotters
plotters.biplot3dnormalized(model,mags,Y.values.astype(float),'classification',dic,k=7,n=7,save=False,path=False)
```

![biplotnormalized](https://github.com/edgor17/Haar-Like-Metric-Learning/assets/87628022/0481569f-74db-4349-aeea-2a8aeda834ae)


Notice that the loadings align with the various body site clusters in the embedding. To more closely examine some of these important Haar-like coordinates we can make a box plot.

```
plotters.boxplot(mags,Y.values,model.coordinates[0:7],dic,dic.keys(),save=False,path=False)
```

![boxplot](https://github.com/edgor17/Haar-Like-Metric-Learning/assets/87628022/76fc8541-9867-4ea5-a60d-92589eca4c8b)


This plot allows us to very clearly see which coordinates can distinguish which body habitats


