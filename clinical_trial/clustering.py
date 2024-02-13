#!/usr/bin/env python
# coding: utf-8

# In[229]:


import nltk
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
## For Stemming
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import os
import re   ## for regular expressions
from mpl_toolkits.mplot3d import Axes3D
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as hc
from IPython.display import HTML


# In[4]:


df = pd.read_csv("cleanerDF.csv")
df.head()


# In[235]:


#data = [df["DesignPrimaryPurpose"],df["BriefSummary"]]
#textdf=pd.DataFrame(data).transpose()
textdf=pd.DataFrame(df["BriefSummary"])
textdf.head()


# In[236]:


A_STEMMER=PorterStemmer()


# In[237]:


def MY_STEMMER(str_input):
    ## Only use letters, no punct, no nums, make lowercase...
    str_input = str_input.replace('-', ' ')
    words = re.sub(r"[^A-Za-z\-]", " ", str_input).lower().split()
    words = [A_STEMMER.stem(word) for word in words] ## Use the Stemmer...
    for word in list(words):
        if len(word) <3:
            words.remove(word)
    return words


# In[238]:


labelsCol = []
contents = []


# In[239]:


len(textdf)


# In[245]:


for i in list(range(len(textdf))):
    #labelsCol.append(textdf["DesignPrimaryPurpose"][i])
    contents.append(textdf["BriefSummary"][i])

for i in list(range(len(textdf))):
    labelsCol.append(df["DesignPrimaryPurpose"][i])


# In[246]:


regex1 = '[a-zA-Z]{3,20}'
MyVectCount=CountVectorizer(input='content',
                        stop_words='english',
                        max_features=500,
                        token_pattern  = regex1
                        )

## Tf-idf vectorizer
MyVectTFIdf=TfidfVectorizer(input='content',
                        stop_words='english',
                        max_features=500,
                        token_pattern  = regex1,
                        )

## Create a CountVectorizer object that you can use with the Stemmer
MyCV_Stem = CountVectorizer(input="content", 
                        stop_words='english', 
                        tokenizer=MY_STEMMER,
                        #token_pattern  = regex1,
                        max_features=500,
                        lowercase=True)


# In[247]:


DTM_Count=MyVectCount.fit_transform(contents)
DTM_TF=MyVectTFIdf.fit_transform(contents)
DTM_stem=MyCV_Stem.fit_transform(contents)


# In[248]:


ColumnNames=MyVectCount.get_feature_names()
#print("The vocab is: ", ColumnNames, "\n\n")
len(ColumnNames)


# In[249]:


ColNamesStem=MyCV_Stem.get_feature_names()
#print("The stemmed vocab is\n", ColNamesStem)
len(ColNamesStem)


# In[250]:


## Use pandas to create data frames
DF_Count=pd.DataFrame(DTM_Count.toarray(),columns=ColumnNames)
DF_TF=pd.DataFrame(DTM_TF.toarray(),columns=ColumnNames)
DF_stem=pd.DataFrame(DTM_stem.toarray(),columns=ColNamesStem)
#print(DF_Count)
#print(DF_TF)
#print(DF_stem)


# In[251]:


DF_Count


# In[252]:


## Now update the row names
MyDict={}
for i in range(0, len(labelsCol)):
    MyDict[i] = labelsCol[i]

#print("MY DICT:", MyDict)


# In[253]:


DF_Count=DF_Count.rename(MyDict, axis="index")


# In[254]:


################################################
##
##         Look at best values for k
##
###################################################

SS_dist = []

values_for_k=range(2,9)

for k_val in values_for_k:
    #print(k_val)
    k_means = KMeans(n_clusters=k_val)
    model = k_means.fit(DF_Count)
    SS_dist.append(k_means.inertia_)
    
print(SS_dist)
print(values_for_k)


# In[255]:


plt.plot(values_for_k, SS_dist, 'bx-')
plt.xlabel('value')
plt.ylabel('Sum of squared distances')
plt.title('Elbow method for optimal k Choice')
plt.show()


# In[256]:


####
# Look at Silhouette
##########################
Sih=[]
Cal=[]
k_range=range(2,9)

for k in k_range:
    k_means_n = KMeans(n_clusters=k)
    model = k_means_n.fit(DF_Count)
    Pred = k_means_n.predict(DF_Count)
    labels_n = k_means_n.labels_
    R1=metrics.silhouette_score(DF_Count, labels_n, metric = 'euclidean')
    R2=metrics.calinski_harabasz_score(DF_Count, labels_n)
    Sih.append(R1)
    Cal.append(R2)


# In[257]:


print(Sih) ## higher is better
print(Cal) ## higher is better


# In[258]:


fig1, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
ax1.plot(k_range,Sih)
ax1.set_title("Silhouette")
ax1.set_xlabel("")
ax2.plot(k_range,Cal)
ax2.set_title("Calinski_Harabasz_Score")
ax2.set_xlabel("k values")


# In[273]:


################################################
##           Let's Cluster........
################################################
kmeans_object_Count = sklearn.cluster.KMeans(n_clusters=5)
kmeans_object_Count.fit(DF_Count)


# In[260]:


# Get cluster assignment labels
labels = kmeans_object_Count.labels_
prediction_kmeans = kmeans_object_Count.predict(DF_Count)
#print(labels)


# In[304]:


labels


# In[298]:


np.unique(labels,return_counts=True)


# In[299]:


df['DesignPrimaryPurpose'].value_counts()


# In[308]:


# Format results as a DataFrame
Myresults = pd.DataFrame([DF_Count.index,labels]).T


# In[320]:


Myresults=Myresults.rename(columns={0:'label',1:'kmean'})


# In[322]:


Myresults.groupby(['label','kmean']).size() # compare label with kmean cluster


# In[288]:


############# ---> ALWAYS USE VIS! ----------
#print(DF_Count)


# In[289]:


# normalize the data 
DF_Count_normalized=(DF_Count - DF_Count.mean()) / DF_Count.std()


# In[290]:


## Instantiated my own copy of PCA
My_pca = PCA(n_components=4)  ## I want the two prin columns


# In[291]:


## Transpose it
DF_Count_normalized=np.transpose(DF_Count_normalized)
My_pca.fit(DF_Count_normalized)


# In[292]:


# Reformat and view results
Comps = pd.DataFrame(My_pca.components_.T,
                        columns=['PC%s' % _ for _ in range(4)],
                        index=DF_Count_normalized.columns
                        )


# In[293]:


Comps


# In[269]:


########################
## Look at 2D PCA clusters
############################################

plt.figure(figsize=(12,12))
plt.scatter(Comps.iloc[:,0], Comps.iloc[:,1], s=100, color="green")

plt.xlabel("PC 1")
plt.ylabel("PC 2")

plt.show()


# In[270]:


#4D PCA

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = Comps.iloc[:,0]
y = Comps.iloc[:,1]
z = Comps.iloc[:,2]
c = Comps.iloc[:,3]

img = ax.scatter(x, y, z, c=c, cmap=plt.hot())
fig.colorbar(img)
plt.show()


# In[271]:


#########################################
##
##  Hierarchical 
##
#########################################


MyHC = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
FIT=MyHC.fit(DF_Count)
HC_labels = MyHC.labels_
print(HC_labels)


# In[272]:


plt.title('Hierarchical Clustering')
dendro = hc.dendrogram((hc.linkage(DF_Count, method ='ward')))


# In[ ]:





# In[ ]:




