#DAN: Nice work with the paragraphs explaining something about the two new methods.
#This needs a header - I can't even tell what your ML problem is without a bit of reverse engineering.
#You'll have another try at the image dataset when you start the NN unit. 
#Nice work getting this done adequately from the wilds.
#Since you have not had time to dig as much as you might have liked into this assignment, you might want 
#to spend 20 minutes going through my code solution for the assignment when you return from Ecuador.
#Grade: S


library(imager)
library(caret)

#image directories
image_dir <- "C:/Users/User/OneDrive - University of Kansas/MLClass/HeyMLbio/UnitCARET/Animals"

#(folders)
categories <- c("cats", "dogs", "snakes")

#image size for resizing
img_height <- 256
img_width <- 256

#function to load and resize images
load_and_resize_image <- function(img_path) {
  img <- load.image(img_path)  # Load image
  img <- resize(img, img_width, img_height)  # Resize image
  img <- grayscale(img)  # Convert to grayscale
  as.numeric(img)  # Convert image to a numeric vector
}

#initialize lists for storing image data and labels
image_data <- list()
labels <- c()

#loop through each category and process images
for (cat in categories) {
  cat_images <- list.files(path = file.path(image_dir, cat), full.names = TRUE)
  cat_data <- lapply(cat_images, load_and_resize_image)  # Load and resize all images
  image_data <- append(image_data, cat_data)  # Append image data
  labels <- append(labels, rep(cat, length(cat_images)))  # Create corresponding labels
}

# Convert image data and labels to a dataframe
image_df <- as.data.frame(do.call(rbind, image_data))  # Combine all image vectors into one dataframe
image_df$label <- as.factor(labels)  # Add labels as a factor

# Check the structure of the dataframe
str(image_df)

#write a CSV file
write.csv(image_df, file = "animals_image_data_raw.csv", row.names = FALSE)

#I couldn't get this to run without losing memory. I want to thin the datasetm but for the sake of time, I figure it would be best for me to just choose a different dataset that I worked with before. I can try on a PC back home or thin the dataset and see if it performs better later, but for now, I will use an existing datsaet.



library(caret)
library(gbm)
library(nnet)
library(e1071)
library(MASS)  # For qda

# Set up train and test data
set.seed(101)
data <- read.csv("Frogs_MFCCs.csv")
shuffled_data <- data[sample(nrow(data)), ]
train_data <- shuffled_data[1:floor(0.75 * nrow(shuffled_data)), ]
test_data <- shuffled_data[(floor(0.75 * nrow(shuffled_data)) + 1):nrow(shuffled_data), ]

# Define models and their tuning grids
model_list <- list(
  gbm = list(
    method = "gbm",
    tuneGrid = expand.grid(
      n.trees = c(50, 100, 150),
      interaction.depth = c(1, 3, 5),
      shrinkage = c(0.01, 0.1),
      n.minobsinnode = 10
    )
  ),
  nnet = list(
    method = "nnet",
    tuneGrid = expand.grid(
      size = c(3, 5, 7),
      decay = c(0.1, 0.5, 1)
    ),
    trace = FALSE  #found online
  ),
  svmRadial = list(
    method = "svmRadial",
    tuneGrid = expand.grid(sigma = 0.05, C = c(1, 2, 3))
  ),
  qda = list(
    method = "qda",
    tuneGrid = NULL  #No tuning parameters for qda
  ),
  multinom = list(
    method = "multinom",
    tuneGrid = NULL  #No tuning parameters for multinom
  )
)

# cross-validation controls
cv_control <- trainControl(method = "cv", number = 5)

#Loop over models, fit each model with cross-validation, and store result:
results <- data.frame(Model = character(), Accuracy = numeric(), Kappa = numeric(), AccuracySD = numeric(), KappaSD = numeric())

for (model_name in names(model_list)) {
  model_info <- model_list[[model_name]]
  set.seed(101)
  trained_model <- tryCatch({
    train(
      Species ~ .,
      data = train_data,
      method = model_info$method,
      tuneGrid = model_info$tuneGrid,
      trControl = cv_control,
      preProcess = c("center", "scale")  #found online to help me scale
    )
  }, error = function(e) {
    message(paste("Error in model:", model_name, "-", e$message))
    NULL
  })
  
  if (is.null(trained_model)) next
  
  #extract best accuracy and Kappa along with their standard deviations
  best_result <- trained_model$results[which.max(trained_model$results$Accuracy), ]
  results <- rbind(results, data.frame(
    Model = model_name,
    Accuracy = best_result$Accuracy,
    Kappa = best_result$Kappa,
    AccuracySD = best_result$AccuracySD,
    KappaSD = best_result$KappaSD
  ))
}

print(results)

#They all compared fairly evenly, but I'd like to explore nnet more as it seems cool to me, mostly from the buzz around the name

expanded_tune_grid_nnet <- expand.grid(
  size = c(3, 5, 7, 10, 15),   #options for hidden layer size
  decay = c(0.01, 0.1, 0.5, 1)  #expanded range for regularization parameter
)

#cross-validation controls for nnet
cv_control_nnet <- trainControl(method = "cv", number = 10) #up the cv

#Train nnet model with the expanded tuning grid
set.seed(101)
optimized_nnet_model <- train(
  Species ~ .,
  data = train_data,
  method = "nnet",
  tuneGrid = expanded_tune_grid_nnet,
  trControl = cv_control_nnet,
  preProcess = c("center", "scale"),  #Apply preprocessing, found online to help me scale
  trace = FALSE  #Suppress training output, not sure what this one does, but found it online
)

#best results for nnet
best_nnet_params <- optimized_nnet_model$bestTune
best_nnet_accuracy <- max(optimized_nnet_model$results$Accuracy)
best_nnet_results <- optimized_nnet_model$results[which.max(optimized_nnet_model$results$Accuracy), ]
print(best_nnet_params)
print(paste("Best Accuracy:", best_nnet_accuracy))
print(best_nnet_results)


library(gplots)

# Set up the PNG device
png("best_nnet_params.png", width = 600, height = 200)

#best parameters for nnet
textplot(best_nnet_params, halign = "center", valign = "center", cex = 1.2)
title("Best Parameters for nnet Model")

library(gridExtra)
library(grid)
png("model_results_summary.png", width = 800, height = 300)

# Convert results to a table grob (graphical object) and display it, also found online
table_grob <- tableGrob(results)
grid.draw(table_grob)

dev.off()
