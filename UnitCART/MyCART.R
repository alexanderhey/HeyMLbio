
#loadin data 
data <- read.csv("Frogs_MFCCs.csv")

#I can't remove any columns, they all seem important

#checking for missing sata
summary(data)
sum(is.na(data))
# no missing data here

#inspecting histograms
hist(data$MFCCs_.8)

#they all look different, I can't say one is better than the others or one needs to be tossed

#CARTING
dataset <- data

#Set random seed
set.seed(101)

#shuffle the rows of the dataset without replacement
shuffled_data <- dataset[sample(dim(dataset)[1], dim(dataset)[1], replace=FALSE),]

#split the shuffled data: 75% for training/validation and 25% for testing
train_data <- shuffled_data[1:floor(.75*(dim(shuffled_data)[1])),]
test_data <- shuffled_data[(floor(.75*(dim(shuffled_data)[1]))+1):dim(shuffled_data)[1],]

#load the rpart lib
library(rpart)

#train a decision tree model on train_data
# ~ is like an equals sign for a formula 
#Species~. uses all predictors to predict 'Species', method="class" specifies classification
model_tree <- rpart(Species ~ ., data = train_data, method = "class")

print(model_tree)

#predict species on the training/validation set
train_predictions <- predict(model_tree, type = "class")

#confusion matrix comparing predicted vs actual species in train_data. not sure what a confusion matrix is
table(train_predictions, train_data$Species)

#calculate the error rate
train_error_rate <- sum(train_predictions != train_data$Species) / dim(train_data)[1]
print(paste("Training Error Rate:", train_error_rate))

#graphical visualization
# Load the rpart.plot library for visualizing the tree
library(rpart.plot)

# Plot the tree
rpart.plot(model_tree)


#Full CART, didn't know how to make it full so I looked at the code 
full_tree<-rpart(Species~.,data=train_data,method="class", control=rpart.control(cp=0,minsplit=1))

#predict species on the training/validation set
train_predictions_full <- predict(full_tree, type = "class")

#confusion matrix comparing predicted vs actual species in train_data. not sure what a confusion matrix is
table(train_predictions_full, train_data$Species)

#calculate the error rate
train_error_rate_full <- sum(train_predictions_full != train_data$Species) / dim(train_data)[1]
print(paste("Training Error Rate:", train_error_rate_full))


#Plot the tree
rpart.plot(full_tree)

#This rpart.plot package seems cool. much easier to read!







#K folding 

#found caret online for k folding, going to try and use it here


library(caret)

#k to use
numberK <- 7

#shuffle the rows of the dataset
shuffled_data <- dataset[sample(dim(dataset)[1], dim(dataset)[1], replace = FALSE),]


#set up cross-validation control using the 'trainControl' function from the caret package
cv_control <- trainControl(method = "cv", number = numberK)
head(cv_control)

#caret's train() function is used for k-fold cross-validation
names(getModelInfo())
#I don't know which model to use, going to use rpart or lm because I saw an example online
kfold_tree_model <- train(Species ~ ., 
                          data = shuffled_data, 
                          method = "rpart", 
                          trControl = cv_control)

#results of cross val
print(kfold_tree_model)

kfold_tree_model$pred

#final small tree
rpart.plot(kfold_tree_model$finalModel)

#out-of-sample error rate for model_tree
model_tree_error_rate <- 1 - max(kfold_tree_model$results$Accuracy)
print(paste("Out-of-sample error rate for small tree:", model_tree_error_rate))



#full_tree cv, added the cp parameters from before 
cv_full_tree <- train(Species ~ ., 
                      data = dataset, 
                      method = "rpart", 
                      trControl = cv_control, 
                      tuneGrid = data.frame(cp = 0))

#cross-validation results for full_tree
print(cv_full_tree)

#out-of-sample error rate for full_tree
full_tree_error_rate <- 1 - max(cv_full_tree$results$Accuracy)
print(paste("Out-of-sample error rate for full_tree:", full_tree_error_rate))


#final full tree
rpart.plot(cv_full_tree$finalModel)

#within sample error agaim
train_predictions_model <- predict(model_tree, type = "class")
table(train_predictions_model, train_data$Species)
#within-sample error rate
within_sample_error_model_tree <- sum(train_predictions_model != train_data$Species) / nrow(train_data)
print(paste("Within-sample error rate for pruned tree:", within_sample_error_model_tree))

#fulltree within error rate
train_predictions_full <- predict(full_tree, type = "class")
table(train_predictions_full, train_data$Species)
#within-sample error rate
within_sample_error_full_tree <- sum(train_predictions_full != train_data$Species) / nrow(train_data)
print(paste("Within-sample error rate for full tree:", within_sample_error_full_tree))


#looks like the out of sample error rate is much higher in the pruned tree, which makes sense because the error rate should be higher with data that wasn't used to train the model
#The out of sample error for the full tree is still higher than within, but still much lower than the pruned tree
#for both within and out of sample error rates, the full tree performed much better 
#according to the plot and the error rates, it looks like my unpruned full tree just did better. Perhaps the data was so complex it would be hard to overfit, and the absence of pruning really helped the model get to a point where it could predict species accurately on both within and outside data
#This is different than in class, where the complex model was overfit to the cancer data, so I'm curious as to why this happened here 















#pruning with r part and using plotcp to visualize 

plotcp(model_tree)
plotcp(full_tree)

#It looks like the best model or best level of pruning is actually no pruning, or the biggest tree 
# As we discussed, it seems like I any value of cp above 0 would actually make the model worse! This is an interesting dataset, and it looks like lower than a cp of 0.007, you get marginal returns in lowering xstd

#Pruned CART, didn't know how to make it full so I looked at the code 
pruned_tree<-rpart(Species~.,data=train_data,method="class", control=rpart.control(cp=0.007,minsplit=1))

#predict species on the training/validation set
train_predictions_pruned <- predict(pruned_tree, type = "class")

#confusion matrix comparing predicted vs actual species in train_data. not sure what a confusion matrix is
table(train_predictions_pruned, train_data$Species)

#calculate the error rate
train_error_rate_full <- sum(train_predictions_pruned != train_data$Species) / dim(train_data)[1]
print(paste("Training Error Rate:", train_error_rate_full))

#instead of an error rate of 0 on the full tree, it has an error rate of .4 percent. A negligable increase perhaps? If we strive for perfection, then the full tree is still better because that was 0


#full_tree cv, added the cp parameters from before 
cv_pruned_tree <- train(Species ~ ., 
                      data = dataset, 
                      method = "rpart", 
                      trControl = cv_control, 
                      tuneGrid = data.frame(cp = 0.007))

print(cv_pruned_tree)
#out-of-sample error rate for pruned_tree
pruned_tree_error_rate <- 1 - max(cv_pruned_tree$results$Accuracy)
print(paste("Out-of-sample error rate for pruned_tree:", pruned_tree_error_rate))

#As expected, out of sample error rate is marginally higher for any cp above 0 



#Day 5 
#Bagging w my full trees
# Load necessary library
library(ipred)
set.seed(101)

#ipredbagg doesn't like characters 
train_data$Species <- as.factor(train_data$Species)

bagging_model <- bagging(Species ~ ., data = train_data, nbagg = 500,
                         control = rpart.control(cp = 0, minsplit = 1))

#predict on training data
bagging_predictions <- predict(bagging_model, train_data)

#confusion matrix and error rate for bagging model on training data
table(bagging_predictions, train_data$Species)
bagging_error_rate <- sum(bagging_predictions != train_data$Species) / nrow(train_data)
print(paste("Training Error Rate (Bagging):", bagging_error_rate))
#still 0 for error rate

#cross-validation for bagging model
set.seed(101)
cv_bagging_model <- train(Species ~ ., 
                          data = dataset, 
                          method = "treebag", 
                          trControl = cv_control)

#cross-validation results for bagging model
print(cv_bagging_model)

#out-of-sample error rate for bagging model
bagging_cv_error_rate <- 1 - max(cv_bagging_model$results$Accuracy)
print(paste("Out-of-sample error rate for Bagging:", bagging_cv_error_rate))
print(paste("Out-of-sample error rate for full_tree:", full_tree_error_rate))
#The bagging model does have a slightly lower out of sample error rate compared to the full_tree

#Random forest
library(randomForest)

# Set up the random forest model with 1000 trees
set.seed(101)
rf_model <- randomForest(Species ~ ., data = train_data, ntree = 1000)

# Predict on training data
rf_predictions <- predict(rf_model, train_data)

#matrix and error rate for random forest model on training data
table(rf_predictions, train_data$Species)
rf_error_rate <- sum(rf_predictions != train_data$Species) / nrow(train_data)
print(paste("Training Error Rate (Random Forest):", rf_error_rate))
cv_rf_model <- train(Species ~ ., 
                     data = dataset, 
                     method = "rf", 
                     trControl = cv_control)
print(cv_rf_model)

rf_cv_error_rate <- 1 - max(cv_rf_model$results$Accuracy)
print(paste("Out-of-sample error rate for Random Forest:", rf_cv_error_rate))

#Random forest has had the best out of sample error rate! cool stuff 
