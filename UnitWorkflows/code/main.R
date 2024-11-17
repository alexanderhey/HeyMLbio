# main.R

if (!dir.exists("results")) dir.create("results")

# Load required libraries
library(rpart)
library(rpart.plot)
library(caret)
library(ipred)
library(randomForest)
library(adabag)
library(xgboost)

source("code/analysis.R")

print("Analysis complete. Check the 'results' folder for outputs.")
