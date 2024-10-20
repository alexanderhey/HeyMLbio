#DAN: The summary in "Results_summary.docx" is just about the minimum. 
#Overall, you've done all the components. You went above and beyond by using caret 
#when we had not (yet) learned that, and that's great. However, you semi-consistently made 
#the sub-optimal choice to use the entire dataset, instead of the training portion
#of the data whenever you used caret to get x-val scores. That means both that you 
#did not follow the best practice of keeping testing data separate, and also it complicates
#comparisons between x-val scores based on the training data and x-val scores based 
#on the whole dataset. That's a non-trivial shortcoming - have a look again through
#your own code to see what I mean. Nevertheless, S+ for going to the extra effort.  


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
#DAN: Nice job adding this, which was not covered.

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
#DAN: Agreed






#K folding 

#found caret online for k folding, going to try and use it here


library(caret)

#k to use
numberK <- 7

#shuffle the rows of the dataset
shuffled_data <- dataset[sample(dim(dataset)[1], dim(dataset)[1], replace = FALSE),]
#DAN: You here start over from the *whole* dataset, reshuffling it again for some 
#reason, but not re-splitting it. This means you are not following the principle
#of keeping the testing data in the vault. 

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
#DAN: You are here using the testing as well as the training data! That's a problem. 

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
                      data = dataset, #DAN: Again using the full for training. 
                      method = "rpart", 
                      trControl = cv_control, 
                      tuneGrid = data.frame(cp = 0.007))
#DAN: The use of cvaret is cool - you went out of your way to learn that and implement it
#before we leared it. But you need to do all the caret fitting on the training data that
#you separated out at the begining instead of on the whole dataset. 

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
#DAN: Now we have switched back to using only the training data. That's good, but it makes
#it difficult to compare to the work above, which used the whole dataset. 

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
print(cv_rf_model$results)

rf_cv_error_rate <- 1 - max(cv_rf_model$results$Accuracy)
print(paste("Out-of-sample error rate for Random Forest:", rf_cv_error_rate))

#Random forest has had the best out of sample error rate! cool stuff 



#Boosting 
library(adabag)

boosted_model <- boosting(Species ~ ., data = train_data, mfinal = 100)
ada_train_predictions <- predict(boosted_model, train_data)$class

#error rate for boosting model on training data
table(ada_train_predictions, train_data$Species)
ada_train_error_rate <- sum(ada_train_predictions != train_data$Species) / nrow(train_data)
print(paste("Training Error Rate (Boosting):", ada_train_error_rate))

#training error rate still 0

#cv with caret as I've been doing, needed a different TuneGrid 
tune_grid <- expand.grid(mfinal = 100,          #boosting iterations
                         maxdepth = 30,         #depth of trees
                         coeflearn = c("Breiman"))  #coefficient learning, not sure which to use, looked around online

#cross-validation
cv_boosting_model <- train(Species ~ ., 
                           data = dataset, 
                           method = "AdaBoost.M1", 
                           trControl = cv_control, 
                           tuneGrid = tune_grid)

# Cross-validation results for boosting model
print(cv_boosting_model)

boosting_cv_error_rate <- 1 - max(cv_boosting_model$results$Accuracy)
print(paste("Out-of-sample error rate for Boosting:", boosting_cv_error_rate))

#Best out of sample error rate yet. Still marginal tho 


#EXTREME Boosting 
library(xgboost)

#putting it in correct format. I looked at Dan's code for this, wasn't sure what it needed from the help doc


#removing genus and family column as they aren't right for the matrix 
train_data <- subset(train_data, select = -c(Family,Genus) )

# Convert the data to a matrix and labels to integer class indices
train_data$Species <- as.numeric(as.factor(train_data$Species))-1
X_train <- as.matrix(train_data[, -which(names(train_data) == "Species")])
y_train <- as.integer(as.factor(train_data$Species)) - 1  #as I understand this will convert factor levels to integers starting from 0

# Set the number of classes based on the number of unique species
num_class <- length(unique(train_data$Species))
m_xgb <- xgboost(data = X_train,
                 label = y_train,
                 max_depth = 6,
                 eta = 0.3,
                 nrounds = 20,
                 objective = "multi:softprob",  # Use multiclass objective, softmax gave me direct class labels which messed up some stuff below 
                 num_class = num_class,        # Number of classes
                 nthread = 2,
                 verbose = 2)

pred_prob <- predict(m_xgb, X_train)
print(pred_prob)
nrow_X_train <- nrow(X_train)
#each row should represent one observation, and each column represents a class probability
pred_matrix <- matrix(pred_prob, nrow = nrow_X_train, ncol = num_class, byrow = TRUE)
pred_labels <- max.col(pred_matrix) - 1 

xgb_train_error_rate <- sum(pred_labels != y_train) / length(y_train)
print(paste("Training Error Rate (XGBoost Multiclass):", xgb_train_error_rate))
#still 0


#cv with caret as I've been doing

tune_grid <- expand.grid(
  nrounds = 20,          #boosting iterations
  max_depth = 6,         #maximum depth of trees
  eta = 0.3,             #learning rate
  gamma = 0,             #min loss reduction
  colsample_bytree = 1,  #subsample ratio of columns when constructing each tree
  min_child_weight = 1,  #funny name. minimum sum of instance weight (hessian) needed in a child. no idea how this regularization works haha
  subsample = 1          #subsample ratio of the training instances
)
#train the XGBoost model
xgb_model <- train(
  Species ~ ., 
  data = train_data,
  method = "xgbTree",
  trControl = cv_control,
  tuneGrid = tune_grid,
)
#DAN: Now you are using the training data again, instead of the whole dataset. That's
#good, but also inconsistent with some of your other model evaluations, complicating 
#comparisons. 

#CV error rate. The other way I had done it in the past didn't work
print(names(xgb_model$results))

#I think because Accuracy cannot be specified in the reggression models, this is not the same metric when used before, but I'll use RMSE 
#RMSE
XGB_cv_error_rate <- min(xgb_model$results$RMSE)

print(paste("Cross-Validation Error Rate (XGBoost Multiclass):", XGB_cv_error_rate))

#It did worse than my other models: 0.0024 error rate



#TESTIN
#The best model was the boosted model, not extreme boosted
# Predict
ada_test_predictions <- predict(boosted_model, test_data)$class

# Confusion matrix comparing predicted vs actual species in test_data
confusion_matrix <- table(ada_test_predictions, test_data$Species)
print("Confusion Matrix for Test Data:")
print(confusion_matrix)

#error rate on the test data
ada_test_error_rate <- sum(ada_test_predictions != test_data$Species) / nrow(test_data)
print(paste("Test Error Rate (Boosting):", ada_test_error_rate))


#0!
#

#summary table
model_names <- c("Full Decision Tree", 
                 "Pruned Decision Tree", 
                 "Bagging", 
                 "Random Forest", 
                 "Boosting",
                 "XGBoost")
cv_scores <- c(full_tree_error_rate,    
               pruned_tree_error_rate,  
               bagging_cv_error_rate,   
               rf_cv_error_rate,        
               boosting_cv_error_rate,  
               XGB_cv_error_rate)       

cv_score_table <- data.frame(Model = model_names, CV_Error_Rate = cv_scores)
print(cv_score_table)

library(gplots)
png("cv_score_table.png", width = 800, height = 400)
textplot(cv_score_table, halign = "center", valign = "top", cex = 1.2)
title("Cross-Validation Error Rates")
dev.off()
