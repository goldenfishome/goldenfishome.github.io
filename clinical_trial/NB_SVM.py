#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Textmining Naive Bayes Example
import nltk
from sklearn import preprocessing
import pandas as pd
import sklearn
import re  
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
## For Stemming
from nltk.tokenize import sent_tokenize, word_tokenize
import os
from sklearn.model_selection import train_test_split
import random as rd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
## conda install python-graphviz
## restart kernel (click the little red x next to the Console)
import graphviz 
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
#from mpl_toolkits.mplot3d import Axes3D 
## conda install python-graphviz
## restart kernel (click the little red x next to the Console)
import graphviz 
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

from sklearn.tree import export_graphviz
#from sklearn.externals.six import StringIO  
from IPython.display import Image  
## conda install pydotplus
import pydotplus
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

#from nltk.stem import WordNetLemmatizer 
#LEMMER = WordNetLemmatizer() 

from nltk.stem.porter import PorterStemmer
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
import string
from sklearn.metrics import plot_confusion_matrix


# In[2]:


STEMMER=PorterStemmer()
#print(STEMMER.stem("fishings"))

# Use NLTK's PorterStemmer in a function
def MY_STEMMER(str_input):   #I like dogs a lot111 !!"
    words = re.sub(r"[^A-Za-z\-]", " ", str_input).lower().split()   # I, like, dogs, a
    words = [STEMMER.stem(w) for w in words]
    return words


# In[3]:


df = pd.read_csv("text.csv")

### Tokenize and Vectorize the Headlines
## Create the list of headlines

HeadlineLIST=[]
LabelLIST=[]

conditions = ["cirrhosis","sepsis","covid","meningococcal"]
word_remove = ["cirrhosis","sepsis","covid","meningococcal","covid-19","bacterial","infection"]

# append the text and labels into list
for nexttest,nextlabel in zip(df["BriefSummary"],df["Condition"]):
    HeadlineLIST.append(nexttest)
    LabelLIST.append(nextlabel)

###### Remove all words match to label away from text list
newTextList = []

for element in HeadlineLIST:
    allWords = element.split(" ") # split the text
    #print(allWords)
    
    # remove the words in topics
    newWordList = []
    for word in allWords:
        word = word.lower()
        #print(word)
        if word in word_remove:
            pass
        else:
            newWordList.append(word)
    
    # join the splited texts
    newWords =" ".join(newWordList)
    newTextList.append(newWords)


#newTextList


# In[4]:


### Instantiate your CV
MyCountV=CountVectorizer(
        input="content",  ## because we have a csv file
        lowercase=True, 
        stop_words = "english",
        tokenizer=MY_STEMMER
        )

DTM = MyCountV.fit_transform(newTextList)
colNames = MyCountV.get_feature_names()
DTM_DF = pd.DataFrame(DTM.toarray(),columns=colNames)


# In[5]:


def RemoveNums(someDF):
    temp=someDF
    myList = []
    for col in temp.columns:
        logical2=str.isalpha(col)
        if (logical2==False):
            myList.append(str(col))
    
    temp.drop(myList,axis=1,inplace=True)
    return temp

finalDF = RemoveNums(DTM_DF)


# In[6]:


labelsDF = pd.DataFrame(LabelLIST,columns=["LABEL"])
origDF = finalDF.iloc[:,:] # save original one w/o copy

# create complete df with labels
dfs = [labelsDF,finalDF]
final_DF_labeled = pd.concat(dfs,axis=1, join='inner')


# In[7]:


final_DF_labeled


# In[8]:


TrainDF1, TestDF1 = train_test_split(final_DF_labeled, test_size=0.3)


# In[9]:


trainLabels = TrainDF1["LABEL"]
TrainDF1 = TrainDF1.drop(["LABEL"],axis=1)
TestLabels = TestDF1["LABEL"]
TestDF1 = TestDF1.drop(["LABEL"],axis=1)


# In[10]:


modelNB = MultinomialNB()

NB = modelNB.fit(TrainDF1,trainLabels)
pred = modelNB.predict(TestDF1)
print(classification_report(TestLabels,pred))
print(accuracy_score(TestLabels,pred))


# In[11]:


cm = confusion_matrix(TestLabels,pred)
print(cm)


# In[12]:


plot_confusion_matrix(NB,TestDF1,TestLabels)


# In[13]:


#############################################
###########  SVM ############################
#############################################
SVM_model = LinearSVC(C=1)
SVM_model.fit(TrainDF1,trainLabels)

plot_confusion_matrix(SVM_model,TestDF1,TestLabels)


# In[14]:


#--------------other kernels
## RBF------------------------------------------
##------------------------------------------------------
SVM_Model2=sklearn.svm.SVC(C=1, kernel='rbf', 
                           verbose=True, gamma="auto")
SVM_Model2.fit(TrainDF1, trainLabels)



plot_confusion_matrix(SVM_Model2,TestDF1,TestLabels)
##-----------------------------------------
## POLY
##_--------------------------------------------------
SVM_Model3=sklearn.svm.SVC(C=100, kernel='poly',degree=3,
                           gamma="auto", verbose=True)

#print(SVM_Model3)
SVM_Model3.fit(TrainDF1, trainLabels)

plot_confusion_matrix(SVM_Model3,TestDF1,TestLabels)


# In[ ]:





# In[30]:


coef = SVM_model.coef_.ravel()
top_positive_coefficients = np.argsort(coef,axis=0)[-10:]
top_negative_coefficients = np.argsort(coef,axis=0)[:10]
top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
plt.figure(figsize=(15, 5))
colors = ["red" if c < 0 else "blue" for c in coef[top_coefficients]]
plt.bar(x= np.arange(2 * 10)  , height=coef[top_coefficients], width=.5,  color=colors)
feature_names = np.array(TrainDF1.columns)
plt.xticks(np.arange(0, (2*10)), feature_names[top_coefficients], rotation=60, ha="right")


# In[53]:


len(SVM_model.coef_[3].ravel())


# In[56]:



def plot_coefficients(MODEL=SVM_model, COLNAMES=TrainDF1.columns, top_features=10):
    ## Model if SVM MUST be SVC, RE: SVM_Model=LinearSVC(C=10)
    for i in range(4):
        coef = MODEL.coef_[i].ravel()
        top_positive_coefficients = np.argsort(coef,axis=0)[-top_features:]
        #print(top_positive_coefficients)
        top_negative_coefficients = np.argsort(coef,axis=0)[:top_features]
        #print(top_negative_coefficients)
        top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
        # create plot
        plt.figure(figsize=(15, 5))
        colors = ["red" if c < 0 else "blue" for c in coef[top_coefficients]]
        plt.bar(  x=  np.arange(2 * top_features)  , height=coef[top_coefficients], width=.5,  color=colors)
        feature_names = np.array(COLNAMES)
        plt.xticks(np.arange(0, (2*top_features)), feature_names[top_coefficients], rotation=60, ha="right")
        plt.title(conditions[i])
        plt.show()


plot_coefficients()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




