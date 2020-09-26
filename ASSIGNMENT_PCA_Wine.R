##### ASSIGNMENT PCA using wine dataset

# Business Problem: Performing principal component analysis for dimension reduction

wine_data = read.csv("E:\\PCA\\wine.csv")
dim(wine_data)
str(wine_data)
table(wine_data$Type)
# we have 178 records and 14 variables. First variable 'Type' has 3 classes. Other
# variables are of numeric type. we will use these 13 numeric vars for PCA

wine <- wine_data[,-1]
head(wine,3)
attach(wine)

summary(wine_data)
# The variables have different ranges, so this will affect our analysis. We will use
# correlation rather than covariance matrix to get PCs.

cor(wine) # there exists correlation among the variables

pca <- princomp(wine,cor = TRUE, scores = TRUE, covmat = NULL)
str(pca)
''' 
gives details about standard deviation, center, number of records, call(function).
Also gives loadings (weights or correlation coefficients), scale and scores.
'''
summary(pca)
'''
  Comp.1    Comp.2    Comp.3    Comp.4     Comp.5     Comp.6    Comp.7     Comp.8     Comp.9 
0.3619885 0.5540634 0.6652997 0.7359900 0.80162293 0.85098116 0.89336795 0.92017544 0.94239698

# 3 PCs are ecplaining 67% of variance, 5 PCs are explaining 80% variance
# 9 PCs are explaining 94% variance, let us do further analysis using them '''

pca$sdev
pca$loadings
pca$scores
pca_scores <- pca$scores
loadings(pca)


sum(pca$loadings[,1]**2) # checking sum of squared weights is 1

plot(pca$scores[,1],pca$scores[,2])
cor(pca$scores[,1],pca$scores[,2])
cor(pca_scores) # now the cor of PCs is 0 . there existed correlation among variables

biplot(pca)
'''
# Bottom axis gives PC1 scores, left axis gives PC2 scores. 
#Top axis gives PC1 loadings and right axis gives PC2 loadings

# the variables close to X-axis influence PC1 the most. from the biplot we see
# that Phenols, proanthicyanins, flavonoids, non-flavonoids and alcalinity
# greatly influence PC1. 
# The variables close to Y-axis influence PC2. They are mainly color and ash

The angle between Phenols and Proanthocyanins is very less, indicating they have
positive correlation. whereas angle btw ash and Phenols is a right angle, so they
are not likely to be correlated. Angle between Hue and Malic is 180 degrees, 
they are negatively correlated.

'''
cor(Phenols,Proanthocyanins) #  0.6124131
cor(Ash,Phenols) # 0.1289795
cor(wine_data$Malic,wine_data$Hue) # -0.5612957

plot(cumsum(pca$sdev*pca$sdev)*100/(sum(pca$sdev*pca$sdev)),type="b")
plot(pca) # first 3 components are explaining much of the variance
plot(pca, type='l') # elbow plot also shows 3 clusters 

############ Hierarchial clustering with 13 variables
wine_normal <- scale(wine)
dist <- dist(wine_normal,method = 'euclidean')
clust_complete <- hclust(dist,method = 'complete')

# Dendrogram: to see how many clusters can be formed
plot(clust_complete,hang = -1,main='Dendrogram with all 13 variables') # showing 2 to 3 clusters
rect.hclust(clust_complete,k=3,border = 'blue')

# forming 3 clusters
groups <- cutree(clust_complete,3)
clust_hier_13vars <- as.matrix(groups)
finaldata <- cbind(clust_hier_13vars,wine_data)
View(aggregate(finaldata[,-c(1,2)],by=list(clust_hier_13vars),FUN=mean)) 

library(RColorBrewer)
library(scales)
palette(alpha(brewer.pal(9,'Set1'), 0.5))
plot(finaldata[,c(3,4)],col=clust_hier_13vars,pch=16,main='Hierarchial clustering with all 13 variables')
# when we visualize the clusters using combination of 2 variables, find overlapping clusters


############ Hierarchial clustering with 3 PCs
wine_3pc <- pca$scores[,1:3]
dist <- dist(wine_3pc,method = 'euclidean')
clust_complete1 <- hclust(dist,method = 'complete')

plot(clust_complete1,hang = -1,main="Dendrogram with 3 PCs") # showing 2 to 3 clusters
rect.hclust(clust_complete1,k=3,border = 'blue') # 3 irregular sized clusters formed

groups <- cutree(clust_complete1,3)
clust_hier_3pc <- as.matrix(groups)
finaldata <- cbind(clust_hier_3pc,finaldata)
View(aggregate(finaldata[,-c(1,2,3)],by=list(clust_hier_3pc),FUN=mean)) 
plot(wine_3pc,col=clust_hier_3pc,pch=16,main='Hierarchial clustering with 3 PCs')
# 2 clusters are dinstinct, but one cluster is overlapping

############ Hierarchial clustering with 5 PCs
wine_5pc <- pca$scores[,1:5]
dist <- dist(wine_5pc,method = 'euclidean')
clust_complete2 <- hclust(dist,method = 'complete')

plot(clust_complete2,hang = -1) # showing 2 to 5 clusters
rect.hclust(clust_complete2,k=3,border = 'blue') # 3 irregular sized clusters formed

groups <- cutree(clust_complete2,3)
clust_hier_5pc <- as.matrix(groups)
finaldata <- cbind(clust_hier_5pc,finaldata)
View(aggregate(finaldata[,-c(1,2,3,4)],by=list(clust_hier_5pc),FUN=mean)) 

plot(wine_5pc,col=clust_hier_5pc,pch=16,main='Hierarchial clustering with 5 PCs')
# 2 clusters are dinstinct, but one cluster is overlapping, similar to 3-PC clustering

############ Hierarchial clustering with 9 PCs
wine_9pc <- pca$scores[,1:9]
dist <- dist(wine_9pc,method = 'euclidean')
clust_complete3 <- hclust(dist,method = 'complete')

plot(clust_complete3,hang = -1) # showing 2 to 5 clusters
rect.hclust(clust_complete3,k=3,border = 'blue') # 3 irregular sized clusters formed

groups <- cutree(clust_complete3,3)
clust_hier_9pc <- as.matrix(groups)
finaldata <- cbind(clust_hier_9pc,finaldata)
View(aggregate(finaldata[,-c(1,2,3,4,5)],by=list(clust_hier_9pc),FUN=mean)) 

plot(wine_9pc,col=clust_hier_9pc,pch=16,main='Hierarchial clustering with 9 PCs')
# 2 clusters are dinstinct, but one cluster is overlapping, similar to 3-PC 
# and 5-PC clustering


## The clusters of hierarchial clustering are not properly separable (neither in
# original data nor using a few PCs).

############### Kmeans clustering using all variables
library(animation)
set.seed(123)
kmeans_13vars <- kmeans(wine_normal,3)
kmeans.ani(wine_normal,3) # shows for 2 variables, but clusters are overlapping
clust_kmeans_13vars <- kmeans_13vars$cluster

finaldata <- cbind(clust_kmeans_13vars,finaldata)
View(aggregate(finaldata[,-c(1:6)],by=list(clust_kmeans_13vars),FUN = mean))

plot(finaldata[,c(7,8)],col=clust_kmeans_13vars,pch=16,main='Kmeans with all 13 variables')
# using 2 variables if plot the clusters, find 3 overlapping clusters  

############### Kmeans clustering using 3 PCs
head(wine_3pc)
kmeans_3pc <- kmeans(wine_3pc,3)
kmeans.ani(wine_3pc,3)
clust_kmeans_3pc <- kmeans_3pc$cluster
finaldata <- cbind(clust_kmeans_3pc,finaldata)
View(aggregate(finaldata[,-c(1:7)],by=list(clust_kmeans_3pc),FUN = mean))

plot(wine_3pc,col=clust_kmeans_3pc,pch=16,main='Kmeans with 3 PCs')
# 3 clear clusters are visible

############### Kmeans clustering using 5 PCs
head(wine_5pc)
kmeans_5pc <- kmeans(wine_5pc,3)
kmeans.ani(wine_5pc,3)
clust_kmeans_5pc <- kmeans_5pc$cluster
finaldata <- cbind(clust_kmeans_5pc,finaldata)
View(aggregate(finaldata[,-c(1:8)],by=list(clust_kmeans_5pc),FUN = mean))

plot(wine_5pc,col=clust_kmeans_5pc,pch=16,main='Kmeans with 5 PCs')
# 3 clear clusters are visible

############### Kmeans clustering using 9 PCs
head(wine_9pc)
kmeans_9pc <- kmeans(wine_9pc,3)
kmeans.ani(wine_9pc,3)
clust_kmeans_9pc <- kmeans_9pc$cluster
finaldata <- cbind(clust_kmeans_9pc,finaldata)
View(aggregate(finaldata[,-c(1:9)],by=list(clust_kmeans_9pc),FUN = mean))

plot(wine_9pc,col=clust_kmeans_9pc,pch=16,main='Kmeans with 9 PCs')
# 3 clear clusters are visible

'''
Though the variance explained by 3 PCs is just 67%, 5-PCs is 80% and
9-PCs is 94%, the clusters formed by using these PCs is similar. '''

'''
CONCLUSIONS

The wine dataset has many variables. We have used principal component analysis
to reduce the dimensions. To know if the principal components will perform better,
we have used clustering on original dataset and then on PC scores.

We have done hierarchial and kmeans clustering using 3,5 and 9 PCs and all
13 variables. The PCs have produced results similar to that of original 
variables.

Principal components have been successful in capturing the essence of the data.
'''




