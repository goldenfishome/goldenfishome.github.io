library(tm)
library(stringr)
library(wordcloud)
library(slam)
library(quanteda)
library(SnowballC)
library(arules)
library(proxy)
library(cluster)
library(stringi)
library(proxy)
library(Matrix)
library(tidytext) # convert DTM to DF
library(plyr) ## for adply
library(ggplot2)
library(factoextra) # for fviz
library(mclust) # for Mclust EM clustering
library(naivebayes)
library(tidyverse)
library(ggplot2)
library(caret)
library(caretEnsemble)
library(psych)
library(Amelia)
library(mice)
library(GGally)
library(e1071)

setwd("C:\\Users\\yujia\\Desktop\\Georgetown\\501\\project\\NB&SVM")
trials <- read.csv("NBCleanedRecord.csv",stringsAsFactors = T,
                   fileEncoding="UTF-8-BOM")
str(trials)

########### Use duration as label
########### split test & train dataset
test_size <- floor(nrow(trials)*0.2)
test_index <- sample(nrow(trials),test_size,replace = F)
test_df <- trials[test_index,]
train_df <- trials[-test_index,]

############# Use status as label
Stest_label <- test_df$OverallStatus
StestDF_noLabel <- test_df[,-which(names(test_df) %in% c("OverallStatus"))]
Strain_label <- train_df$OverallStatus
StrainDF_noLabel <- train_df[,-which(names(train_df)%in%c("OverallStatus"))]

############ Naive Bayes
NB_S <- naive_bayes(StrainDF_noLabel,
                    Strain_label,
                    laplace = 1)

Sy_pred <- predict(NB_S,StestDF_noLabel)
table(Sy_pred,Stest_label)  
table(Stest_label)

## reasonable for "Active, not recruiting" status, since it only have 2 rows to train. 
## Higher accuracy than the prediction of length


########## Cross validation & feature importance
########### for status
Sx <- StrainDF_noLabel
Sy <- Strain_label
modelS = train(Sx,Sy,"nb")
modelS$results

PredS <- predict(modelS,StestDF_noLabel)
table(PredS,Stest_label)
impS <- varImp(modelS)
plot(impS)
(accuracy <-sum(PredS == Stest_label)/length(Stest_label))

