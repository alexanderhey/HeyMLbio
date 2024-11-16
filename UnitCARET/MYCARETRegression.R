
library(caret)
library(glmnet)  #ridge and lasso regression
library(earth)   #MARS
library(elasticnet) #Elastic Net

train_data <- read.csv("df_train.csv")
test_data <- read.csv("df_test.csv")

#lookoing at data
str(train_data)
summary(train_data)
sum(is.na(train_data))

#Data
train_data <- train_data[sapply(train_data, is.numeric)]
test_data <- test_data[sapply(test_data, is.numeric)]

#target variable is home price
target_variable <- "price"

#model list
model_list <- list(
  ridge = list(
    method = "ridge",
    tuneGrid = expand.grid(lambda = seq(0.01, 1, by = 0.1)),
    trControl = trainControl(method = "cv", number = 5)
  ),
  lasso = list(
    method = "lasso",
    tuneGrid = expand.grid(fraction = seq(0.1, 1, by = 0.1)),
    trControl = trainControl(method = "cv", number = 5)
  ),
  earth = list(
    method = "earth",
    tuneGrid = expand.grid(degree = 1:2, nprune = seq(2, 10, by = 2)),
    trControl = trainControl(method = "cv", number = 5)
  ),
  enet = list(
    method = "enet",
    tuneGrid = expand.grid(lambda = seq(0.01, 1, by = 0.1), fraction = seq(0.1, 1, by = 0.1)),
    trControl = trainControl(method = "cv", number = 5)
  )
)

#Fit Each Model and Collect Results
results <- data.frame(Model = character(), RMSE = numeric(), Rsquared = numeric(), RMSESD = numeric(), RsquaredSD = numeric())

for (model_name in names(model_list)) {
  model_info <- model_list[[model_name]]
  set.seed(101)
  trained_model <- train(
    as.formula(paste(target_variable, "~ .")),
    data = train_data,
    method = model_info$method,
    tuneGrid = model_info$tuneGrid,
    trControl = model_info$trControl,
    preProcess = c("center", "scale")  # Standardize predictors
  )
  
  #extract the best RMSE and R-squared values
  best_result <- trained_model$results[which.min(trained_model$results$RMSE), ]
  results <- rbind(results, data.frame(
    Model = model_name,
    RMSE = best_result$RMSE,
    Rsquared = best_result$Rsquared,
    RMSESD = best_result$RMSESD,
    RsquaredSD = best_result$RsquaredSD
  ))
}

print(results)

#earth had the best R^2 and thus captured the most variance in the data. I will expand that one below

expanded_tune_grid_earth <- expand.grid(
  degree = 1:3,           #expanding degree to explore more complexity
  nprune = seq(2, 30, by = 2)  # max number of terms in the model
)

#Re-train earth with expanded grid
set.seed(101)
optimized_earth_model <- train(
  as.formula(paste(target_variable, "~ .")),
  data = train_data,
  method = "earth",
  tuneGrid = expanded_tune_grid_earth,
  trControl = trainControl(method = "cv", number = 10),
  preProcess = c("center", "scale")
)

#best tuning parameters and results
print(optimized_earth_model$bestTune)
print(optimized_earth_model$results[which.min(optimized_earth_model$results$RMSE), ])

#This one had a better RMSE from the function above, so I think it has been tuned correctly


png("model_results_summary.png", width = 800, height = 300)


table_grob <- tableGrob(results)
grid.draw(table_grob)
