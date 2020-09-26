############ ASSIGNMENT PCA using wine dataset

# Business Problem: To perform dimension reduction using principal component analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

# Data set details
wine_data = pd.read_csv("E:/PCA/wine.csv")
wine_data.shape # 178 obs, 14 variables
wine_data.columns
''' 'Type', 'Alcohol', 'Malic', 'Ash', 'Alcalinity', 'Magnesium', 'Phenols',
       'Flavanoids', 'Nonflavanoids', 'Proanthocyanins', 'Color', 'Hue',
       'Dilution', 'Proline' '''
wine_data.info() # all are continuous data

from collections import Counter
Counter(wine_data.Type)
# Type is categorical data with 3 classes

wine_data.describe()

# let us remove variable 'Type' 
wine = wine_data.iloc[:,1:]
wine.head()

# normalize the data
wine_normal = scale(wine)

# Model building
pca = PCA()
pca_scores = pca.fit_transform(wine_normal)
pca_scores = pd.DataFrame(pca_scores)
pca_scores.columns=['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10','PC11','PC12','PC13']
pca_scores.shape 
pca_scores.head()
# 178, 13  same number of Principal components as the variables

pca_comp = pd.DataFrame([pca.components_[0],pca.components_[1],pca.components_[2],
                         pca.components_[3],pca.components_[4],pca.components_[5],
                         pca.components_[6],pca.components_[7],pca.components_[8],
                         pca.components_[9],pca.components_[10],pca.components_[11],
                         pca.components_[12]])

pca_var = pd.Series(['Alcohol', 'Malic', 'Ash', 'Alcalinity', 'Magnesium', 'Phenols',
       'Flavanoids', 'Nonflavanoids', 'Proanthocyanins', 'Color', 'Hue',
       'Dilution', 'Proline'])

# pca_comp = pca_comp.transpose()
pca_comp = pd.concat([pca_var,pca_comp],axis=1)
pca_comp.columns = ['Variables','PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10','PC11','PC12','PC13']
pca_comp

# to calculate pca scores (using standr variabes and PC) in excel
pca_comp.to_excel(r'E:\PCA\pca.xlsx')
wine_normal = pd.DataFrame(wine_normal)
wine_normal.columns=['Alcohol', 'Malic', 'Ash', 'Alcalinity', 'Magnesium', 'Phenols',
       'Flavanoids', 'Nonflavanoids', 'Proanthocyanins', 'Color', 'Hue',
       'Dilution', 'Proline']
wine_normal = wine_normal.add_suffix('_std')
wine_normal.to_excel(r'E:\PCA\wine_normal.xlsx')

# Amount of variance explained by each PC
explained_var = pca.explained_variance_ratio_ # explained variance amount
explained_var_percent = np.round(explained_var,decimals=4)*100 # explained variance percentage
cum_var = np.cumsum(explained_var) # cumulative variance
cum_var_percent = np.cumsum(np.round(explained_var,decimals = 4)*100) # cumulative variance percentage

variance_explained_df = pd.DataFrame([explained_var,explained_var_percent,cum_var,cum_var_percent])
a = pd.Series(['explained_var','explained_var_percent','cum_var','cum_var_percent'])
variance_explained_df = pd.concat([a,variance_explained_df],axis=1)
variance_explained_df.columns = ['Variance','PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10','PC11','PC12','PC13']
variance_explained_df # details of explained and cumulative variance

# Variance plot for PCA components obtained 
plt.plot(cum_var,color="red")

# plot between Prinicipal Components
x = pca_scores[:,0]
pca_scores[:,1]
# x = np.array(pca_values[:,0])
y = pca_scores[:,1]
z = pca_scores[:,2]
plt.plot(x,y,'bo')
plt.plot(x,z,'bo')

''' Now we have obtained Principal components. First 9 PCs have explained 94% 
of variance of the original variables . So let us choose these 9 PCs. Let us run 
clustering machine learning algorithm first using all 13 variables and then using
these 9 PCs. Let us see if the reduced dimensions will give same result as 
original variables. '''

#################### Hierarchial Clustering using 13 Variables
wine_normal.head(3)
import scipy.cluster.hierarchy as sch

clust_dend = sch.linkage(wine_normal, method='complete', metric = 'euclidean')
%matplotlib qt
sch.dendrogram(clust_dend); # 3 or 4 clusters are clearly visible

from sklearn.cluster import AgglomerativeClustering
hclust_complete = AgglomerativeClustering(n_clusters=3,linkage='complete',affinity='euclidean').fit(wine_normal)
cluster_labels = pd.Series(hclust_complete.labels_)
wine_data.insert(0,'clust_complete',cluster_labels)
aggr_complete= wine_data.iloc[:,2:].groupby(wine_data.clust_complete).mean()

######################## Hierarchial clustering using 9 PC
pca_scores.columns
pca_scores_new = pca_scores.iloc[:,0:9] # selecting first 9 PCs (94% variance explained)
pca_scores_new.columns


clust_dend_pca = sch.linkage(pca_scores_new, method='complete', metric = 'euclidean')
%matplotlib qt
sch.dendrogram(clust_dend_pca); # 3 or 4 clusters are clearly visible

'''
PC = 9, then 3 or 4 clusters
PC =10, then 2 or 5 clusters
'''

hclust_complete_pc9 = AgglomerativeClustering(n_clusters=3,linkage='complete',affinity='euclidean').fit(pca_scores_new)
cluster_labels = pd.Series(hclust_complete_pc9.labels_)
wine_data.insert(0,'clust_complete_PC9',cluster_labels)
aggr_complete_PC9= wine_data.iloc[:,3:].groupby(wine_data.clust_complete_PC9).mean()
wine_data.columns

#plotting clusters 
import seaborn as sns
pca_scores_new.insert(0,'clust_complete_PC9',pd.Series(hclust_complete_pc9.labels_))
sns.lmplot(data=pca_scores_new, x='PC1', y='PC2',hue='clust_complete_PC9',
           fit_reg=False, legend=True, legend_out=True)
# only 2 clusters clearly separated, 3rd cluster samples(dots) are overlapping


pd.crosstab(wine_data.clust_complete, wine_data.clust_complete_PC9)
'''
clust_complete_PC9   0   1   2
clust_complete                
0                   60   9   0
1                   46  11   1
2                    2   3  46  '''

pd.crosstab(wine_data.clust_complete, wine_data.Type)
'''
Type             1   2   3
clust_complete            
0               51  18   0
1                8  50   0
2                0   3  48 '''

pd.crosstab(wine_data.clust_complete_PC9, wine_data.Type)
'''
Type                 1   2   3
clust_complete_PC9            
0                   59  48   1
1                    0  20   3
2                    0   3  44 '''
''' the clustering is not very good when use hierarchial clustering.
when compare aggregate means for 13 variables and 9 PCs, the means are 
different '''

#################### Kmeans Clustering using 13 Variables

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(wine_normal)
cluster_labels = pd.Series(kmeans.labels_)
wine_data.insert(0,'clust_kmeans',cluster_labels)
aggr_kmeans= wine_data.iloc[:,4:].groupby(wine_data.clust_kmeans).mean()


#################### Kmeans Clustering using first 9 PCs

kmeans_pc9 = KMeans(n_clusters=3, random_state=0)
kmeans_pc9.fit(pca_scores_new) # using 9 PCs
cluster_labels = pd.Series(kmeans_pc9.labels_)
wine_data.insert(0,'clust_kmeans_pc9',cluster_labels)
aggr_kmeans_pc9= wine_data.iloc[:,5:].groupby(wine_data.clust_kmeans_pc9).mean()

#plotting clusters 
pca_scores_new.insert(0,'clust_kmeans_pc9',cluster_labels)
%matplotlib qt
clus1 = sns.lmplot(data=pca_scores_new, x='PC1', y='PC2', hue='clust_kmeans_pc9', 
                   fit_reg=False, legend=True, legend_out=True)
# 3 clear clusters are visible


# Creating a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('E:\PCA\pca.xlsx', engine='xlsxwriter')

# Wriing each aggregate output to a different worksheet.
aggr_complete.to_excel(writer, sheet_name='Sheet1')
aggr_complete_PC9.to_excel(writer, sheet_name='Sheet2')
aggr_kmeans.to_excel(writer, sheet_name='Sheet3')
aggr_kmeans_pc9.to_excel(writer,sheet_name='Sheet4')

writer.save() # closing writer and saving the file

import os
os.getcwd()

pd.crosstab(wine_data.clust_kmeans, wine_data.Type)
'''
Type           1   2   3
clust_kmeans            
0             59   3   0
1              0  65   0
2              0   3  48'''
pd.crosstab(wine_data.clust_kmeans_pc9, wine_data.Type)
# same result as above
pd.crosstab(wine_data.clust_kmeans,wine_data.clust_kmeans_pc9)
'''
clust_kmeans_pc9   0   1   2
clust_kmeans                
0                 62   0   0
1                  0  65   0
2                  0   0  51 '''
''' the aggregate means are same when we compared kmeans clsuters
using all 13 variables and 9 PCs '''


''' 
CONCLUSIONS

The wine dataset has 13 variables. For further analysis, using more
variables may reduce the performance. Hence we have performed principal
component analysis. The variance explained by 3 PCs was just 66% and 
5 PCs was just 80%. So we have used 9 Pcs which explains 94% variance.

Next we have performed clustering analysis to check the performance of
9 PCs. The PCs obtained have given good results with kmeans than with 
hierarchial clustering. 

A few principal components have been useful in explaining the variance
 in the dataset.
'''


