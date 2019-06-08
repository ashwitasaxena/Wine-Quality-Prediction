library(dplyr)

setwd("C:/university of cincinnati/SPRING SEMESTER 2019/BANA 7047 DATA MINING II/Assignments/Final Project/Wine Quality")
wine <- read.csv(file = "winequality-white.csv", sep = ";")

wine$response <- as.ordered(ifelse(test = (wine$quality <= 4), 0, ifelse((wine$quality >= 7), 2 ,1)))
table(wine$response)

wine$density <- NULL
wine$quality <- NULL

wine_good <- wine[wine$response == 2, ]
wine_ok <- wine[wine$response == 1, ]
wine_bad <- wine[wine$response == 0, ]

set.seed(04276151)
index_good <- sample(nrow(wine_good), nrow(wine_good) * 0.8)
wine_good_train <- wine_good[index_good, ]
wine_good_test <- wine_good[-index_good, ]

set.seed(04276151)
index_bad <- sample(nrow(wine_bad), nrow(wine_bad) * 0.8)
wine_bad_train <- wine_bad[index_bad, ]
wine_bad_test <- wine_bad[-index_bad, ]

set.seed(04276151)
index_ok <- sample(nrow(wine_ok), nrow(wine_ok) * 0.8)
wine_ok_train <- wine_ok[index_ok, ]
wine_ok_test <- wine_ok[-index_ok, ]

wine.train <- rbind(wine_good_train, wine_ok_train, wine_bad_train)
wine.test <- rbind(wine_bad_test, wine_ok_test, wine_good_test)

#shuffle data
set.seed(04276151)
index <- sample(nrow(wine.train))
wine.train <- wine.train[index,]
index <- sample(nrow(wine.test))
wine.test <- wine.test[index,]


#######
#Rpart#
#######
library(rpart)
library(rpart.plot)

#build tree
wine.rpart <- rpart(formula = as.ordered(response) ~ ., data = wine.train, method = "class")

#view and plot
wine.rpart
prp(wine.rpart, extra = 1)

#make in-sample predictions
preds <- predict(wine.rpart, newdata = wine.train, type = "class")

#confusion matrix
table(wine.train$response, preds, dnn = c("True", "Pred"))

#in-sample error
mean(as.character(wine.train$response) != preds)

#make out-of-sample predictions
preds <- predict(wine.rpart, newdata = wine.test, type = "class")

#confusion matrix
table(wine.test$response, preds, dnn = c("True", "Pred"))

#out-of-sample error
mean(as.character(wine.test$response) != preds)


###############
#Random forest#
###############
library(randomForest)
wine.rf <- randomForest(as.ordered(response)~., data = wine.train)
wine.rf

varImpPlot(wine.rf, main = "Variable Importance")

wine.rf$importance

plot(wine.rf, lwd=rep(2,3))

#in-sample
wine.rf.pred <- predict(wine.rf, type = "class")

table(wine.train$response,wine.rf.pred, dnn=c("Observed","Predicted"))

#in-sample error
mean(as.character(wine.train$response) != wine.rf.pred)

#out-of-sample
wine.rf.pred <- predict(wine.rf, newdata = wine.test, type = "class")

table(wine.test$response,wine.rf.pred, dnn=c("Observed","Predicted"))

#error
mean(as.character(wine.test$response) != wine.rf.pred)

################
#Neural Network#
################
library(nnet)
library(neuralnet)
library(NeuralNetTools)

#can only do one layer, size = number of neurons
y <- class.ind(wine.train$response)
x <- as.matrix(wine.train[,1:10])
wine.nnet <- nnet(x, y, size=5, maxit=500, softmax = TRUE)

#####neuralnettools
library(caret)
library(e1071)
par(mfrow=c(1,1))
#wine.train$response %>% factor %>% unique
wine.nnet2<-train(factor(response)~.,data=wine.train,method="nnet")
print(wine.nnet2)
plot(wine.nnet2)
plotnet(wine.nnet2$finalModel,y_names="response")
title("Graphical Representation of Neural Network")

#in-sample
winetrain<- wine.train[,1:10]
preds <- predict(wine.nnet2, winetrain, type = "raw")

table(wine.train$response,preds, dnn=c("Observed","Predicted"))

mean(as.character(wine.train$response) != preds)

#out-of-sample
preds <- predict(wine.nnet, newdata = wine.test[,1:10], type = "class")

table(wine.test$response,preds, dnn=c("Observed","Predicted"))

mean(as.character(wine.test$response) != preds)


#############################
#Ordinal Logistic Regression#
#############################

library(MASS)
wine.train$response <- as.ordered(wine.train$response)
wine.polr <- polr(formula = response ~ ., data = wine.train, Hess = TRUE)
summary(wine.polr)

#in-sample
wine.predict <- predict(object = wine.polr, wine.train, type = "class")
table(wine.train$response, wine.predict, dnn = c("True", "Pred"))
mean(as.character(wine.train$response) != as.character(wine.predict))

#out-of-sample
wine.predict <- predict(object = wine.polr, wine.test, type = "class")
head(wine.predict)
summary(wine.predict)
table(wine.test$response, wine.predict)
mean(as.character(wine.test$response) != as.character(wine.predict))

names(wine.train)
table(wine.train$response)



#####
#GAM#
#####

table(wine.train$response)
# For GAM the response whould be numeric

library(mgcv)

wine.gam <- gam(as.numeric(response) ~ fixed.acidity + s(volatile.acidity) + s(citric.acid) + 
                  s(residual.sugar) + s(chlorides) + s(free.sulfur.dioxide) + total.sulfur.dioxide + 
                  pH + sulphates + s(alcohol), family = ocat(R = 3), 
                data = wine.train, method = "REML")


summary(wine.gam)
plot(wine.gam, pages = 1, shade = 2, scale = 0)

gam.check(wine.gam)
#in-sample
wine.gam.pred <- predict(wine.gam, wine.train, type = "response", se = TRUE)
colnames(wine.gam.pred$fit) <- c("0", "1", "2")

head(wine.gam.pred$fit)

#Below code is to get the highest prob among the one's and convert that in to the
maxpos = apply(wine.gam.pred$fit, 1, FUN = function(v)which(v == max(v)))
table(maxpos)
maxpos <- maxpos - 1
#xtabs(~wine.train$response + maxpos)
tab = table(wine.train$response, maxpos)
tab

#in-sample
mean(wine.train$response != maxpos)

#out-of-sample
wine.gam.pred <- predict(wine.gam, wine.test, type = "response", se = TRUE)
colnames(wine.gam.pred$fit) <- c("0", "1", "2")

head(wine.gam.pred$fit)

#Below code is to get the highest prob among the one's and convert that in to the
maxpos = apply(wine.gam.pred$fit, 1, FUN = function(v)which(v == max(v)))
maxpos <- maxpos - 1
#xtabs(~wine.test$response + maxpos)
tab = table(wine.test$response, maxpos)
tab

mean(wine.test$response != maxpos)


#####################
# Gradient Boosting #
#####################
library(gbm)
table(wine.test$response)
wine.gbm <- gbm(formula = response ~ ., data = wine.train, distribution = 'multinomial',
                n.trees = 200, interaction.depth = 4, shrinkage = 0.005 )
summary(wine.gbm)
#n-sample
pred.gbm <- predict(object = wine.gbm, newdata = wine.train, type = "response", n.trees = 200)

col_gbm <- c("0","1","2")
colnames(pred.gbm) <- col_gbm

maxpos_gbm = apply(pred.gbm, 1, which.max)
table(maxpos_gbm)
maxpos_gbm <- maxpos_gbm - 1
#xtabs(~wine.train$response + maxpos_gbm)
tab = table(wine.train$response, maxpos_gbm)
tab


mean(as.character(wine.train$response) != as.character(maxpos_gbm))


#out-of-sample
pred.gbm <- predict(object = wine.gbm, newdata = wine.test, type = "response", n.trees = 200)
col_gbm <- c("0","1","2")
colnames(pred.gbm) <- col_gbm

maxpos_gbm = apply(pred.gbm, 1, which.max)
maxpos_gbm <- maxpos_gbm - 1
#xtabs(~wine.test$response + maxpos_gbm)
tab = table(wine.test$response, maxpos_gbm)
tab


mean(as.character(wine.test$response) != as.character(maxpos_gbm))
