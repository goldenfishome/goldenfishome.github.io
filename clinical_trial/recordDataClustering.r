library(stats)  ## for dist
library(NbClust)
library(cluster)
library(mclust)
library(amap)  ## for Kmeans (notice the cap K)
library(factoextra) ## for cluster vis, silhouette, etc.
library(purrr)
library(stylo)  ## for dist.cosine
library(philentropy)  ## for distance() which offers 46 metrics
library(SnowballC)
library(caTools)
library(dplyr)
library(textstem)
library(stringr)
library(wordcloud)
library(tm) 

originalDF <- read.csv("HepatitisCdata.csv",na.strings=c(""," ","NA"))
sum(is.na(originalDF))

## fill nan value with mean
for (i in 1:ncol(originalDF)){
  originalDF[is.na(originalDF[,i]),i] <- mean(originalDF[,i],na.rm=T)
}

str(originalDF)
# change datatype of columns
originalDF$Category <- as.factor(originalDF$Category)

# save labels of data set
labels <- originalDF$Category

# create numeric only data frame with label
numDF_label <- subset(originalDF,select = -c(X,Sex))

# create numeric only data frame
numDF <- subset(numDF_label,select = -Category)

# sample dataset
a <- read.csv("clusterData.csv")
dist1 <- dist(a,method="minkowski", p=2)

str(numDF)
#### use at least three different distance metrics and compare the results.
## use Euclidean, or Cosine Sim, or Minkowski with p = 1
M2_Eucl <- dist(numDF,method="minkowski", p=2)  # Euclidean distance
M1_Man <- dist(numDF,method="manhattan") # manhanttan distance
CosSim <- stylo::dist.cosine(as.matrix(numDF)) # cosine similarity distance

(M2_Eucl[1:10])
(M1_Man[1:10])
(CosSim[1:10])

# determine the number of clusters.
## use Elbow, Silhouette, and Gap-Stat to illustrate visually (and in your discussion) the ideal value of k in k means.

kmeans_3D_1<-NbClust::NbClust(numDF, min.nc=2, max.nc=5, method="kmeans")
table(kmeans_3D_1$Best.n[1,])

barplot(table(kmeans_3D_1$Best.n[1,]), 
        xlab="Numer of Clusters", ylab="",
        main="Number of Clusters")

fviz_nbclust(numDF, method = "silhouette", FUN = hcut, k.max = 6)
WSS <- fviz_nbclust(numDF, method = "wss",FUN = hcut,  k.max = 6) + ggtitle("WSS:Elbow")
WSS
# silhouette show k at 2, while elbow show at 3


#### Perform k- means (at least 3 values of k) and hierarchical clustering.
k <-4
kmeansResult1 <- kmeans(numDF, k) ## uses Euclidean
kmeansResult1$centers

#print(kmeansResult1)

## Compare to the labels
# discussion 
table(labels,kmeansResult1$cluster)
summary(kmeansResult1)

## Place results in a tbale with the original data
clusterDF <- cbind(numDF_label, cluster = kmeansResult1$cluster)

## This is the size (the number of points in) each cluster
# Cluster size
kmeansResult1$size
table(labels)
# able to distinguish from healthy people

# list of cluster assignments
o = order(kmeansResult1$cluster)

####MANY visualizations for the k-means clusters and the hierarchical clustering (dendrograms).
#- Challenge yourself and have a 3D vis that can be rotated
fviz_cluster(kmeansResult1, numDF, main="Potential Hepatitis C Groups")
# Euclidian

clusplot(numDF,kmeansResult1$cluster,color = T,
         shade = T,labels = 2,lines = 0,
         main="Potential Hepatitis C Groups")


#######################################################
##          Hierarchical CLustering
#######################################################
hc <- hclust(M2_Eucl, method = "average")
plot(hc,cex=0.9, hang=-1, 
     main = "Cluster of Potential Hepatitis C Groups")
rect.hclust(hc, k=4)



