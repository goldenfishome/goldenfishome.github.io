library(rpart)   ## FOR Decision Trees
library(rattle)  ## FOR Decision Tree Vis
library(rpart.plot)
library(RColorBrewer)
library(Cairo)
library(network)
library(ggplot2)
library(wordcloud)
library(slam)
library(quanteda)
library(proxy)
library(stringr)
library(textmineR)
library(igraph)
library(caret)

setwd("C:/Users/yujia/Desktop/Georgetown/501/project/DT")
df <- read.csv("DT_recordDF.csv",stringsAsFactors = T)
names(df)[1] <- "OverallStatus"
str(df)

apply(df,2,table)

GoPlot <- function(x){
  G <- ggplot(data=df,aes(.data[[x]],y=""))+
    geom_bar(stat="identity",aes(fill=.data[[x]]))
  
  return(G)
}

# create visualization for counts of variables in each column
#lapply(names(df),function(x) GoPlot(x))

Gini = function(x){
  return(1-sum(x^2))
}

entropy = function(x){
  return(-sum(x*log2(x)))
}


###################################### 
##split the data into train & test 
(dataSize = nrow(df))
(train_size = floor(dataSize*3/4))
test_size = dataSize-train_size

set.seed(1234)
# sample random rows with given size
train_sample_index <- sample(nrow(df),train_size
                             ,replace = F)

train_set <- df[train_sample_index,] # create training data set
table(train_set$OverallStatus)

test_set <- df[-train_sample_index,] # create training data set
table(test_set$OverallStatus)

##################
# Remove the labels from test set
test_label <- test_set$OverallStatus
test_set <- test_set[-1]

#######################
## Decision tress
str(train_set)


#### Tree DT
DT <- rpart(train_set$OverallStatus~.,
            data = train_set,
            method = "class")
summary(DT)
plotcp(DT)

# visualisation
DT_prediction = predict(DT,test_set,
                        type="class")
table(DT_prediction,test_label) # confusion matrix
fancyRpartPlot(DT,cex=0.6)
DT$variable.importance

#### Tree DT2
DT2<-rpart(train_set$OverallStatus ~ DesignInterventionModel+DesignPrimaryPurpose, 
           data = train_set, 
           method="class",minsplit=2)
## The small cp the larger the tree if cp is too small you have overfitting
summary(DT2)

DT2_prediction = predict(DT2,test_set,
                         type="class")
table(DT2_prediction,test_label) # confusion matrix

fancyRpartPlot(DT2,cex=0.8)

#### Tree DT3
DT3<-rpart(train_set$OverallStatus ~ EnrollmentCount+AgeGroups, 
           data = train_set, 
           method="class",minsplit=2)

DT3_prediction = predict(DT3,test_set,
                         type="class")
table(DT3_prediction,test_label) # confusion matrix

fancyRpartPlot(DT3,cex=0.6)
