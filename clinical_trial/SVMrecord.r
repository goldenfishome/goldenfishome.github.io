library(e1071)
library(mlr)
library(caret)
library(ggplot2)
library(MASS) 

setwd("C:\\Users\\yujia\\Desktop\\Georgetown\\501\\project\\NB&SVM")
trials <- read.csv("SVM_DF.csv",fileEncoding="UTF-8-BOM",
                   stringsAsFactors = T)

test_size <- floor(nrow(trials)*0.2)
test_index <- sample(nrow(trials),test_size,replace = F)
test_df <- trials[test_index,]
train_df <- trials[-test_index,]

Stest_label <- test_df$OverallStatus
StestDF_noLabel <- test_df[,-which(names(test_df) %in% c("OverallStatus"))]
Strain_label <- train_df$OverallStatus
StrainDF_noLabel <- train_df[,-which(names(train_df)%in%c("OverallStatus"))]

plot(trials)

library(lattice)
xyplot(DurationDays~AdverseEffectsorDeath,
       groups = OverallStatus,
       data=trials)

qplot(trials$EnrollmentCount,
      trials$DurationDays,
      data=trials,
      color=trials$OverallStatus)

###### SVM
SVM_fit_P <- svm(OverallStatus~.,data=train_df,
                 kernel="polynomial",cost=.1,
                 scale=FALSE)
print(SVM_fit_P)
pred <- predict(SVM_fit_P,StestDF_noLabel,
                type="class")

(accuracy_p <-sum(pred == Stest_label)/length(Stest_label))

(table(pred,Stest_label))
plot(SVM_fit_P,data = train_df,DurationDays~AdverseEffectsorDeath)
plot(SVM_fit_P,data = train_df,DurationDays~EnrollmentCount)
plot(SVM_fit_P,data = train_df,AdverseEffectsorDeath~EnrollmentCount)
###############################
## Linear Kernel...
SVM_fit_L <- svm(OverallStatus~.,data = train_df,
                 kernal="linear",cost=.1,
                 scale = FALSE)
#print(SVM_fit_L)
pred_L <- predict(SVM_fit_L,StestDF_noLabel,
                  type="class")

(L_table<-table(pred_L,Stest_label))

plot(SVM_fit_L,data = train_df,DurationDays~AdverseEffectsorDeath)
(accuracy_l <-sum(pred_L == Stest_label)/length(Stest_label))
####################################
## Radial Kernel...
SVM_fit_R <- svm(OverallStatus~., data=train_df, 
                 kernel="radial", cost=.1, 
                 scale=FALSE)
print(SVM_fit_R)
pred_R <- predict(SVM_fit_R, StestDF_noLabel, type="class")
(R_table<-table(pred_R, Stest_label))
(accuracy_R <-sum(pred_R == Stest_label)/length(Stest_label))
