
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


#Full CART
full_tree<-rpart(Species~.,data=train_data,method="class", control=rpart.control(cp=0,minsplit=1))

#predict species on the training/validation set
train_predictions_full <- predict(full_tree, type = "class")

#confusion matrix comparing predicted vs actual species in train_data. not sure what a confusion matrix is
table(train_predictions_full, train_data$Species)

#calculate the error rate
train_error_rate_full <- sum(train_predictions_full != train_data$Species) / dim(train_data)[1]
print(paste("Training Error Rate:", train_error_rate_full))


# Plot the tree
rpart.plot(full_tree)

#This rpart.plot package seems cool. much easier to read!