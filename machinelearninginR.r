library("caret")
#preprocessing 
library(GGally)

#visualize variables 
ggpairs(data=iris,aes(color=Species))

#Missing values 
#remove NA
data("airquality")
summary (airquality)
new_airquality <- na.omit(airquality)
summary (new_airquality)

#Set NA = Mean value 
NA2mean <- function(x) replace(x, is.na(x), mean(x, na.rm = TRUE))
new_airquality2 <-replace(airquality, TRUE, lapply(airquality, NA2mean))
summary (new_airquality2)

#Imputation
PreImputeBag <-preProcess(airquality,method="bagImpute")
DataImputeBag <-predict(PreImputeBag,airquality)
summary (DataImputeBag)

#Data Partitioning 
ind1 <- createDataPartition(y=iris$Species,p=0.6,list=FALSE,times=1)
  #List = false prevents results being listed
  #p = 0.6: percentage of data split for training
  #times=1, one split 
training <-iris[ind1,]
testing <-iris[-ind1,]

#k-folds and cross validation
#k = groups to split
fitControl <- trainControl(method="cv", number=10) #train the model
model <- train(Species~., data=training, trControl=fitControl, method="lda")
print(model)
predict1 <- predict(model,testing)

#classification model: evaluation of prediction 
#predict species of Iris based on 4 variables
train_inputs=training[,1:4]
train_outputs=training[,5]
model <- train(train_inputs, train_outputs, method="lda")
#use trained model on test set
predictions <-predict(model,testing)
#evaluate accuracy
mean(predictions==testing$Species)

#Confusion Matrix
confusionMatrix (predictions,testing$Species)
#visualize the confusion matrix
library(reshape2)
cm <- confusionMatrix(predictions,testing$Species)
cm_df <- melt(cm$table)
ggplot(cm_df, aes(x = Prediction, y = Reference, fill = value)) + 
  geom_raster() + scale_fill_distiller(palette = "Spectral")

#Regression and Classification Models
predictions <- rep (0, dim(mtcars)[1])
for (i in 1:dim(mtcars[1])) {
  training <-mtcars[-i,]
  testing <- mtcars[i,]
  train_inputs=training[,2:11]
  train_outputs=training[,1]
  test_inputs=testing[,2:11]
  model <- train(train_inputs, train_outputs, method="lm")
  predictions[i] <- predict(model, test_inputs)
}
#visualizing
qplot (predictions, mtcars$mpg)
#Pearson's correlation efficient
cor (predictions, mtcars$mpg)
cor.test (predictions, mtcars$mpg)

#Training with the regression method
#Pre-process the data
library(caret)
data(airquality)
set.seed(123)
#Use bagging approach for missing values 
preImputeBag <- preProcess(airquality,method="bagImpute")
airquality_imp <-predict(PreImputeBag, airquality)
indT<- createDataPartition(y=airquality_imp$Ozone,p=0.6,list=FALSE)
training <- airquality_imp[indT,]
testing <- airquality_imp[-indT,]
#Build model based on one predictor
ModFit <- train(Ozone~Temp,data=training,
                preProcess=c("center", "scale"),
                method="lm")
summary(ModFit$finalModel)
#Apply and analyze output
prediction <- predict(ModFit, testing)
cor.test(prediction,testing$Ozone)
#Build model based on two predictors
modFit2 <- train(Ozone~Solar.R+Wind+Temp,data=training, 
                 preProcess=c("center", "scale"),
                 method="lm")
summary(modFit2$finalModel)
prediction2 <- predict(modFit2,testing)
cor.test(prediction2,testing$Ozone)
#stepwise linear regression
modFit_SLR <- train(Ozone~Solar.R+Wind+Temp,data=training,method="lmStepAIC")
summary(modFit_SLR$finalModel)
prediction_SLR <- predict(modFit_SLR,testing)
cor.test(prediction_SLR,testing$Ozone)
postResample(prediction_SLR,testing$Ozone)
#Polynomial Regression
modFit_poly <-train(Ozone~poly(Solar.R,3)+poly(Wind,3)+poly(Temp,3),data=training,
                    preProcess=c("center","scale"),
                    method="lm")
summary(modFit_poly$finalModel)
prediction_poly <-predict(modFit_poly,testing)
cor.test(prediction_poly,testing$Ozone)
#Principal Component Regression 
library(pls) 
modFit_PCR <- train(Ozone~Solar.R+Wind+Temp,data=training,method="pcr")
summary(modFit_PCR$finalModel)
prediction_PCR <-predict(modFit_PCR,testing)
cor.test(prediction_PCR,testing$Ozone)

#Categorical Output: Logistic Regression
library(kernlab)
library(ROCR)
data(spam)
names(spam)
#Train model
indTrain <- createDataPartition(y=spam$type,p=0.6,list = FALSE)
training <- spam[indTrain,]
testing <-spam[-indTrain,]
ModFit_glm <-train(type~.,data=training,method="glm")
summary(ModFit_glm$finalModel)
#Predictions
predictions <-predict(ModFit_glm,testing)
confusionMatrix(predictions, testing$type) 
#Plot ROC, compute AUC
pred_prob <-predict(ModFit_glm,testing,type="prob")
head(pred_prob)
data_roc <-data.frame(pred_prob = pred_prob[,'spam'],
                      actual_label = ifelse(testing$type == 'spam', 1, 0))
roc <-prediction(predictions = data_roc$pred_prob,
                 labels = data_roc$actual_label)
plot(performance(roc, "tpr", "fpr"))
abline(0, 1, lty = 2) 
auc <-performance(roc, measure = "auc")

#Classification with Decision Boundaries 
library(caret)
data(iris)
set.seed(123)
indT <- createDataPartition(y=iris$Species,p=0.6,list=FALSE)
training <- iris[indT,]
testing  <- iris[-indT,]

ModFit_KNN <- train(Species~.,training,method="knn",tuneGrid = expand.grid(k = 1:25))

ggplot(ModFit_KNN$results,aes(k,Accuracy))+
  geom_point(color="blue")+
  labs(title=paste("Optimum K is ",ModFit_KNN$bestTune),
       y="Accuracy")

predict_KNN<- predict(ModFit_KNN,newdata=testing)
confusionMatrix(testing$Species,predict_KNN)

#Training a model using tree-based model 
library(caret)
data(iris)
set.seed(123)
indT <-createDataPartition(y=iris$Species,p=0.6,list=FALSE)
training <-iris[indT,]
testing <-iris[-indT,] 
#train using method=rpart, gini algorithm 
ModFit_rpart <- train(Species~.,data=training,method="rpart",
                      parms = list(split = "gini"))
#can be replaced by any other splitting algorithm
library(rattle)
fancyRpartPlot(ModFit_rpart$finalModel)
predict_rpart <- predict(ModFit_rpart,testing)
confusionMatrix(predict_rpart, testing$Species)
testing$PredRight <- predict_rpart==testing$Species
ggplot(testing,aes(x=Petal.Width,y=Petal.Length))+
  geom_point(aes(col=PredRight))

library(randomForest)
ModFit_rf <- train(Species~.,data=training,method="rf",prox=TRUE)
predict_rf <- predict(ModFit_rf,testing)
confusionMatrix(predict_rf, testing$Species)
testing$PredRight <- predict_rf==testing$Species
ggplot(testing,aes(x=Petal.Width,y=Petal.Length)) + 
  geom_point(aes(col=PredRight)) 

#Training Machine Learning model using Ensemble Approach
ModFit_bag <- train(as.factor(Species) ~ .,data=training,
                    method="treebag",
                    importance=TRUE)
predict_bag <- predict(ModFit_bag,testing)
confusionMatrix(predict_bag, testing$Species)
plot(varImp(ModFit_bag))

#Unsupervised Learning
install.packages("factoextra")
library(ggplot2)
library(factoextra)
library(purrr)
data(iris)
ggplot(iris,aes(x=Sepal.Length,y=Petal.Width)) + 
  geom_point(aes(color=Species))
set.seed(123)
km <- kmeans(iris[,3:4],3,nstart=20)

table(km$cluster,iris$Species)
fviz_cluster(km,data=iris[,3:4]) 
fviz_nbclust(iris[,3:4], kmeans, method = "wss")

#Neural Networks
library(caret)
library(neuralnet)
#split the data
datain <- mtcars
set.seed(123) 
#split for training and testing
indT <- createDataPartition(y=datain$mpg,p=06,list=FALSE)
training <- datain[indT,]
testing <- dataom[-indT,]
#scale data
smax <- apply(training,2,max)
smin <- apply(training,2,min)
trainNN <-as.data.frame(scale(training,center=smin,scale=smax-smin))
testNN <-as.data.frame(scale(testing,center=smin,scale=smax-smin))

library(caret)
library(neuralnet)
datain <- mtcars
set.seed(123)
#Split training/testing
indT <- createDataPartition(y=datain$mpg,p=0.6,list=FALSE)
training <- datain[indT,]
testing  <- datain[-indT,]
#scale the data set
smax <- apply(training,2,max)
smin <- apply(training,2,min)
trainNN <- as.data.frame(scale(training,center=smin,scale=smax-smin))
testNN <- as.data.frame(scale(testing,center=smin,scale=smax-smin))
#fit Neural Network: 1 hidden layer, 10 neurons w back propagation 
set.seed(123)
ModNN <- neuralnet(mpg~cyl+disp+hp+drat+wt+qsec+carb,trainNN, hidden=10,linear.output = T)
plot(ModNN)
#Predict with neural network
predictNN <- compute(ModNN,testNN[,c(2:7,11)])
predictmpg <- predictNN$net.result*(smax-smin[1]+smin[1])
postResample(testing$mpg,predictmpg)
