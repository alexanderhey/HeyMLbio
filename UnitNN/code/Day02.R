

#going to add more images to see if that helps the models 
# Image directories
project_dir <- getwd()  # Get the current working directory (project root)
image_dir <- file.path(project_dir, "data", "Animals")

# Folders/categories
categories <- c("cats", "dogs", "snakes")

# Image size for resizing
img_height <- 256
img_width <- 256

#Function to load and resize images
load_and_resize_image <- function(img_path) {
  img <- load.image(img_path)  #Load image
  img <- resize(img, img_width, img_height)  #Resize image
  img <- grayscale(img)  #Convert to grayscale
  as.numeric(img)  #Convert image to a numeric vector
}

#Initialize lists for storing image data and labels
image_data <- list()
labels <- c()

#Loop through each category and process
for (cat in categories) {
  cat_images <- list.files(path = file.path(image_dir, cat), full.names = TRUE)
  
  #Limit to 500 images per category
  cat_images <- head(cat_images, 500)
  
  #Load and process images
  cat_data <- lapply(cat_images, load_and_resize_image)  # Load and resize all images
  image_data <- append(image_data, cat_data)  # Append image data
  labels <- append(labels, rep(cat, length(cat_images)))  # Create corresponding labels
}

# Convert image data and labels to a dataframe
image_df <- as.data.frame(do.call(rbind, image_data))  #Combine all image vectors into one dataframe
image_df$label <- as.factor(labels)  
# Check the structure of the dataframe
str(image_df)
data <- image_df

#Extract predictors (image data) and response variable (labels)
x <- as.matrix(data[, -ncol(data)])  # All columns except the last
y <- as.numeric(as.factor(data$label)) - 1  #Convert labels to numeric starting at 0

#Normalize the predictors
x <- x / 255

#Split into training and testing sets
set.seed(123)
train_indices <- sample(1:nrow(x), size = 0.8 * nrow(x))  # 80% for training
x_train <- x[train_indices, ]
y_train <- y[train_indices]
x_test <- x[-train_indices, ]
y_test <- y[-train_indices]

#Flatten the 4D data into 2D. found this online as well.
x_train_flat <- array_reshape(x_train, c(dim(x_train)[1], img_height * img_width))
x_test_flat <- array_reshape(x_test, c(dim(x_test)[1], img_height * img_width))

#Define the fully connected model
inputs <- layer_input(shape = c(img_height * img_width))  #Updated input shape for flattened data
outputs <- inputs %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dense(units = length(unique(y_train)), activation = 'softmax')

model <- keras_model(inputs = inputs, outputs = outputs)

#Compile the model
model %>% compile(
  optimizer = optimizer_adam(),
  loss = 'sparse_categorical_crossentropy',
  metrics = c('accuracy')
)

#Train the model
history <- model %>% fit(
  x_train_flat, y_train,
  epochs = 100,
  batch_size = 32,
  validation_split = 0.2
)

#well that still did terrible. still can't get much past 50% validation accuracy. perhaps changing the loss function will be better...

#I was having trouble here with the shape of my categorical y data with the new loss function, so I learned I had to make them one-hot encoded labels. 

#One-hot encode the labels for categorical_crossentropy
y_train_one_hot <- to_categorical(y_train, num_classes = length(unique(y_train)))
y_test_one_hot <- to_categorical(y_test, num_classes = length(unique(y_test)))


#Define a new model with a different loss function
inputs2 <- layer_input(shape = c(img_height * img_width)) 
outputs2 <- inputs2 %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dense(units = length(unique(y_train)), activation = 'softmax')

model2 <- keras_model(inputs = inputs2, outputs = outputs2)


#Compile the second model
model2 %>% compile(
  optimizer = optimizer_adam(),
  loss = 'categorical_crossentropy', #new loss function
  metrics = c('accuracy')
)

#Train the second model
history2 <- model2 %>% fit(
  x_train_flat, y_train_one_hot,  
  epochs = 100,
  batch_size = 32,
  validation_split = 0.2
)

#Define the results directory
results_dir <- file.path(getwd(), "results")  
if (!dir.exists(results_dir)) dir.create(results_dir)

#Function to save accuracy plot with informative names, and echoing stuff for the user so when they source it it isn't confusing 
save_accuracy_plot <- function(history, filename) {
  #Extract accuracy and validation accuracy
  train_acc <- history$metrics$accuracy
  val_acc <- history$metrics$val_accuracy
  epochs <- seq_along(train_acc)
  
  #Plot
  plot(epochs, train_acc, type = "l", col = "blue", ylim = c(0, 1),
       xlab = "Epoch", ylab = "Accuracy", main = paste("Accuracy Plot:", filename))
  lines(epochs, val_acc, col = "red")
  legend("bottomright", legend = c("Training", "Validation"),
         col = c("blue", "red"), lty = 1)
  
  # Save plot
  full_filename <- file.path(results_dir, paste0(filename, "_accuracy_plot.png"))  # EDIT HERE
  dev.copy(png, full_filename)
  dev.off()
  message("Saved plot as: ", full_filename)
}

#save
save_accuracy_plot(history, "sparse_categorical_crossentropy")
save_accuracy_plot(history2, "new_model_categorical_crossentropy")

actual_classes <- y_test
#convert one-hot encoded y_test to class indices
actual_classes <- apply(y_test_one_hot, 1, which.max) - 1
cat("Predicted classes:", head(predicted_classes), "\n")
cat("Actual classes:", head(actual_classes), "\n")

#some informing stats
correctly_classified_indices <- which(predicted_classes == actual_classes)
misclassified_indices <- which(predicted_classes != actual_classes)

cat("Number of correctly classified images:", length(correctly_classified_indices), "\n")
cat("Number of misclassified images:", length(misclassified_indices), "\n")

# Function to save classified images. found help with this online because it was hard to point it to a real directory 
save_classified_image <- function(index, type) {
  #Reverse normalization (scale pixel values back to 0-255)
  image_data <- x_test_flat[index, ] * 255
  
  #Reshape the flattened data into its original dimensions (256x256)
  image_matrix <- matrix(image_data, nrow = 256, ncol = 256, byrow = TRUE)
  
  #Predicted and actual labels
  predicted_label <- categories[predicted_classes[index] + 1]
  actual_label <- categories[actual_classes[index] + 1]
  
  #filename construction.
  filename <- file.path(results_dir, paste0(type, "_image_", index, ".png"))  # EDIT HERE
  
  #Save image
  png(filename)
  image(t(image_matrix)[, ncol(image_matrix):1], col = gray.colors(256),
        main = paste(type, "\nPredicted:", predicted_label, "\nActual:", actual_label))
  dev.off()
  message("Saved ", type, " image as: ", filename)
}

#Save examples of correctly classified images
cat("Saving correctly classified images...\n")
for (i in 1:5) {  #Adjust the numbers here to save more or fewer images
  save_classified_image(correctly_classified_indices[i], "correctly_classified")
}

#Save examples misclassified images
cat("Saving misclassified images...\n")
for (i in 1:5) {  
  save_classified_image(misclassified_indices[i], "misclassified")
}
