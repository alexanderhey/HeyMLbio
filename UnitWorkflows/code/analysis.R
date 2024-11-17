data <- read.csv("data/Frogs_MFCCs.csv")

#missing data
summary(data)
sum(is.na(data))

#Prepare data
set.seed(101)
shuffled_data <- data[sample(nrow(data)), ]
train_data <- shuffled_data[1:floor(0.75 * nrow(shuffled_data)), ]
test_data <- shuffled_data[(floor(0.75 * nrow(shuffled_data)) + 1):nrow(shuffled_data), ]

#train decision tree
model_tree <- rpart(Species ~ ., data = train_data, method = "class")
train_predictions <- predict(model_tree, train_data, type = "class")
train_error_rate <- sum(train_predictions != train_data$Species) / nrow(train_data)

#predict on test data for validation
test_predictions <- predict(model_tree, test_data, type = "class")
test_error_rate <- sum(test_predictions != test_data$Species) / nrow(test_data)

#save tree plot
png("results/decision_tree.png")
rpart.plot(model_tree, main="Decision Tree")
dev.off()

#train a pruned tree model
pruned_tree <- rpart(Species ~ ., data = train_data, method = "class", control = rpart.control(cp = 0.007, minsplit = 1))
train_predictions_pruned <- predict(pruned_tree, train_data, type = "class")
train_error_rate_pruned <- sum(train_predictions_pruned != train_data$Species) / nrow(train_data)

test_predictions_pruned <- predict(pruned_tree, test_data, type = "class")
test_error_rate_pruned <- sum(test_predictions_pruned != test_data$Species) / nrow(test_data)

# pruned tree plot
png("results/pruned_tree.png")
rpart.plot(pruned_tree, main="Pruned Decision Tree")
dev.off()

#cross-validation
cv_control <- trainControl(method = "cv", number = 7)
cv_tree_model <- train(Species ~ ., data = train_data, method = "rpart", trControl = cv_control)
cv_error_rate <- 1 - max(cv_tree_model$results$Accuracy)

png("results/cv_tree.png")
rpart.plot(cv_tree_model$finalModel, main="Cross-Validated Decision Tree")
dev.off()

#results as CSV for reference
results <- data.frame(
  Method = c("Unpruned Tree (Train)", "Unpruned Tree (Test)", 
             "Pruned Tree (Train)", "Pruned Tree (Test)",
             "Cross-Validated Tree"),
  ErrorRate = c(train_error_rate, test_error_rate,
                train_error_rate_pruned, test_error_rate_pruned,
                cv_error_rate)
)
write.csv(results, "results/error_rates.csv", row.names = FALSE)
