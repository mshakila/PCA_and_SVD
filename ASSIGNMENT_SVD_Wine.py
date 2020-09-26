################### SVD (Singular-Value Decomposition) on wine dataset 
###### comparing SVD, PCA performance on wine dataset using Kmeans clustering 

# Business Poblem: dimensionality reduction using SVD, PCA. Then using kmeans
# clustering to know how svd and PCA have performed 

import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
import seaborn as sns
from numpy import array

wine_data = pd.read_csv("E:/PCA/wine.csv")
wine = wine_data.iloc[:,1:] # removing 'Type' which is categorical having 3 classes
wine_normal = scale(wine) # nromalizing the data to remove influence of large values

################## SVD using sklearn library's TruncatedSVD

from sklearn.decomposition import TruncatedSVD
# using data wine_normal 13 variables and 178 rows
wine_normal[0:4,]

# svd                                                          
svd = TruncatedSVD(n_components=3)
svd.fit(wine_normal)

# transforming
svd_3col=svd.transform(wine_normal)
svd_3col[0:4,]

############## SVD using scipy library's svd

from numpy import diag
from numpy import zeros
from scipy.linalg import svd

# defining matrix A of (original matrix)
A = wine_normal.copy()
A.shape  # (178, 13)
A[0:4,0:4] 

''' As per SVD we can divide any matrix A into its constituent parts OR that 
A is a combo of 3 matrices. 
A = U . Sigma . V^T 
 '''

# Singular-value decomposition
U, s, VT = svd(A)
U.shape # (178, 178)
s.shape  # 13 elements
VT.shape # (13, 13)

# create m x n Sigma matrix
Sigma = zeros((A.shape[0], A.shape[1]))
Sigma.shape # (178, 13) with all zero elements

# populate Sigma with n x n diagonal matrix
Sigma[:A.shape[1], :A.shape[1]] = diag(s)
Sigma.shape # (178, 13) with diagonal filled by elements of 's' others zero

# selecting 3 components 
'''
The top 6 diagonal values of Sigma are 28,21,16,12,12,10. 
The top 3 singular values are 28,21 and 16. So we will be selecting
3 components '''

n_elements = 3
Sigma = Sigma[:, :n_elements] # 178,3
VT = VT[:n_elements, :]
VT.shape # (3,13)

# reconstruct
# B = U.dot(Sigma.dot(VT))
# print(B)

# transform
T = U.dot(Sigma) # 178,3
T[0:4,]

# checking T is same as T1
# T1 = A.dot(VT.T)
# T1[0:4,]
# both T and T1 are same, both are equivalent transforms of original matrix

# let us compare the datasets obtained from scipy library's svd and sklearn's
# TruncatedSVD
T[0:5]
svd_3col[0:5]
# same values but signs different. 
'''
SVD suffers from this problem called “sign indeterminacy”, which means the 
sign of the components_ and the output from transform depend on the 
algorithm and random state. From trial and error we will choose the good fit.
'''

################## PCA
from sklearn.decomposition import PCA

pca = PCA()
pca_scores = pca.fit_transform(wine_normal)
pca_scores = pd.DataFrame(pca_scores)
pca_scores.columns=['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10','PC11','PC12','PC13']

# Amount of variance explained by each PC
explained_var = pca.explained_variance_ratio_ # explained variance amount
cum_var_percent = np.cumsum(np.round(explained_var,decimals = 4)*100) # cumulative variance percentage
# 36.20, 55.41, 66.53, 73.60, 80.16, 85.10, 89.34, 92.02, 94.24,
 #  96.17, 97.91, 99.21, 100.01

# selecting first 3 PCs (67% variance explained)
pca_scores_new = pca_scores.iloc[:,0:3] 
pca_scores_new.columns

#################### Kmeans Clustering using 13 Variables

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(wine_normal)
cluster_labels = pd.Series(kmeans.labels_)
wine_data.insert(0,'clust_kmeans',cluster_labels)
aggr_kmeans= wine_data.iloc[:,2:].groupby(wine_data.clust_kmeans).mean()

clus_plot = sns.lmplot(data=wine_data, x='Alcohol',y='Malic', hue='clust_kmeans', 
                       fit_reg=False, legend=True, legend_out=True)
# since many variables in dataset and can plot using only 2 variables, 
# cannot find distinct clusters

############ kmeans clustering using svd (3 components)
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(svd_3col)
cluster_labels = pd.Series(kmeans.labels_)
wine_data.insert(0,'clust_kmeans_svd3',cluster_labels)
aggr_kmeans_svd3= wine_data.iloc[:,3:].groupby(wine_data.clust_kmeans_svd3).mean()

'''
when we compare the aggregate means, of clusters formed by original variables
and svd (3 components) we find cluster-2 means of both is same.
the clusters 2 and 3 are interchaged and their means are almost same.
'''
svd_3col_df = pd.DataFrame(svd_3col)
svd_3col_df.columns = ['svd1','svd2','svd3']
svd_3col_df.insert(0,'clust_kmeans_svd3',cluster_labels)

# plotting
%matplotlib qt
clus_plot = sns.lmplot(data=svd_3col_df, x='svd1',y='svd2', hue='clust_kmeans_svd3', 
                       fit_reg=False, legend=True, legend_out=True)
# 3 clear clusters are visible

####################### kmeans using PCA 3 components
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(pca_scores_new)
cluster_labels = pd.Series(kmeans.labels_)
wine_data.insert(0,'clust_kmeans_pca3',cluster_labels)
aggr_kmeans_pca3= wine_data.iloc[:,4:].groupby(wine_data.clust_kmeans_pca3).mean()
# pca and svd have given exact results when we look into aggregate means

pca_scores_new.insert(0,'clust_kmeans_pca3',cluster_labels)

%matplotlib qt
clus_plot = sns.lmplot(data=pca_scores_new, x='PC1',y='PC2', hue='clust_kmeans_pca3', 
                       fit_reg=False, legend=True, legend_out=True)
# 3 clear clusters are visible which are similar to svd-clusters 

'''
CONCLUSIONS

We have performed SVD and PCA for reducing dimensions in wine dataset.

We have obtained 3 components using the original 13 variables. To know how 
good the components are, we have performed clustering. Kmeans clustering with
original variables and the svd and pca components have given similar results. 

The svd and pca components have been successful in capturing the essence 
(variance) of the data. '''


