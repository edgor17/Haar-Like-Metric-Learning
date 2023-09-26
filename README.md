# Adaptive Haar-like Metric Learning

The Adaptive Haar-like metric learns a data dependent set of Haar-coordinates and associated weights to best represent a given dataset. 

## Installation
```
git clone https://github.com/edgor17/Haar-like-Metric-Learning
cd WHAT
pip install .
```

# Tutorial for a Classification Type Dataset

We first demonstrate our metric on a dataset from "Bacterial community variation in human body habitats across space and time." This data is publically available on QIITA (ID 449). For this example we will show how the Haar-like metric can distinguish between different body habitats that the samples were taken from. 

## Preprocessing
First we need to generate a Haar-like basis for the phylogeny associated with this dataset. In this case, Greengenes 97 was the reference phylogeny. We use ete toolkit to process .nwk trees. Note that this step only needs to be done once for any phylogenetic tree.

```
from ete3 import Tree
from preproccesing import Haar_Build
tree=Tree("97_otus_unannotated.tree",format=1)
haarlike=Haar_Build(tree)
```
Next we map the given feature table onto the reference phylogeny and sort by the label of interest. Here we use host_body_habitat. mags is the result of applying the Haar-like transformation to the feature table. dic is a dictionary containing an integer mapping of the unique metadatalabels of interest, this will be useful for plotting. 

```
[X,Y,mags,dic]=PreProcess(featuretable,metadata,'host_body_habitat','classification',tree,haarlike)
```

## Training 
Now we are ready to train our metric. The important parameters here are s, nsamples, and min_samples_leaf. s is the number of Haar-like coordinates to recover during matching pursuit. nsamples is the number of samples to use for training. min_samples_leaf is a random forest parameter to set the minimum number of points that form a bin. For classification problems we always set this to 1. 

```
model = AdaptiveHaarLike(labeltype='classification')
model.fit(X,Y,s=10,mags,nsamples=len(X),min_samples_leaf=1)
```

At this point we can compare the Random Forest Gram matrix to our learned Gram matrix

```
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 5))
axes[0].imshow(model.rfgram.todense(),vmin=0,vmax=.3)
axes[1].imshow(model.Reconstruct(mags,10),vmin=0,vmax=.3)
axes[0].xaxis.set_visible(False)
axes[1].xaxis.set_visible(False)
fig.tight_layout()
print(list(dic.keys()))
```
![rfbodysite](https://user-images.githubusercontent.com/87628022/236347131-c721aaa4-0d4a-458d-b408-8d63f94897b2.png)

['skin', 'external auditory canal', 'feces', 'hair', 'oral cavity', 'nostril', 'urine']

## Results
Having learned a Haar-like Gram matrix, we are now ready to plot the embedding. Here we set n=7 to plot the 7 dominant Haar-like coordinate loadings. 

```
biplot3d(model,mags,Y.values.astype(float),'classification',dic,s=10,n=7)
```

![biplot3dbodysites](https://user-images.githubusercontent.com/87628022/236350152-60495ec1-9a1f-43aa-8479-63baa314772f.png)


Notice that the loadings align with the various body site clusters in the embedding. To more closely examine some of these important Haar-like coordinates we can make a box plot.

```
boxplot(mags,Y.values.astype(float),model.coordinates[0:7],dic,list(dic.keys()))
```
![boxplotbodysites](https://user-images.githubusercontent.com/87628022/236348013-d0f4d4de-6148-4521-bbc5-fe5e1049946f.png)

This plot allows us to very clearly see which coordinates can distinguish which body habitats










# Tutorial for a Regression Type Dataset

Next we consider microbial mat samples from "Trade-offs between microbiome diversity and productivity in a stratified microbial mat." This data is publically available on QIITA (ID 10481). For our analysis we consider the depth at which a sample was taken.

## Preprocessing
We map the given feature table onto the reference phylogeny and sort by the label of interest. Here we use matdepthcryosectionmm. mags is the result of applying the Haar-like transformation to the feature table. For regression problems dic is unneccsary and set to None.

```
[X,Y,mags,dic]=PreProcess(featuretable,metadata,'matdepthcryosectionmm','regression',tree,haarlike)
```

## Training 
For regression problems the learned metric can be somewhat sensitive to the min_samples_leaf parameter. Typically, we set this higher than 1 so that the original random forest is able to recover relationships between nearby points. 

```
model = AdaptiveHaarLike(labeltype='regression')
model.fit(X,Y,s=10,mags,nsamples=len(X),min_samples_leaf=15)
```

At this point we can compare the Random Forest Gram matrix to our learned Gram matrix

```
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 5))
axes[0].imshow(model.rfgram.todense(),vmin=0,vmax=.3)
axes[1].imshow(model.Reconstruct(mags,10),vmin=0,vmax=.3)
axes[0].xaxis.set_visible(False)
axes[1].xaxis.set_visible(False)
fig.tight_layout()
```
![comparerfgramregression](https://user-images.githubusercontent.com/87628022/236315805-c109b16e-765b-492b-8a79-6e72cbf396df.png)


The Adaptive Haar-like metric struggles to reconstruct some of the middle samples, but overall recovers the same diagonal pattern as the Random Forest. 

## Results
Here we set n=3 to plot the 3 dominant Haar-coordinate loadings. 

```
biplot2d(model,mags,Y.values.astype(float),'regression',dic,10,3)
```

![biplotregression](https://user-images.githubusercontent.com/87628022/236375154-9726a6ab-1767-47cb-98b9-8f236c9b6391.png)


This embedding has a clear gradient with respect to sample depth, also the loadings align well with this gradient. Lets take a look at each of these.

```
magplot(mags,Y.values.astype(float),model.coordinates[0:3],'depth (mm)')
```
![hccoordsregression](https://user-images.githubusercontent.com/87628022/236319182-aaab9ba9-25d3-4a5c-a46f-8c82fcce6ab3.png)
