#processing the images better for a CNN. I saw the way you processed them in class and had to try it myself, as it looks way better than the way I was doing it before and captures more data

project_dir <- getwd()  
image_dir <- file.path(project_dir, "data", "Animals")

categories <- c("cats", "dogs", "snakes")  

img_height <- 256
img_width <- 256

load_and_preprocess_image <- function(img_path) {
  img <- load.image(img_path)  #DAN: This is an imager function, so you need a library(imager)
  img <- resize(img, img_width, img_height)  #resize to dimensions. Not sure if I need this since kaggle said they were all 256 by 256
  img_array <- as.array(img)  #Convert to array
  img_array <- array(img_array, dim = c(img_height, img_width, 3))  #Add channel dimension for RGB
  return(img_array)
}

image_data <- list()
labels <- c()

#process image loop
for (i in seq_along(categories)) {
  cat <- categories[i]
  cat_images <- list.files(path = file.path(image_dir, cat), full.names = TRUE)
  cat_data <- lapply(cat_images, load_and_preprocess_image)  # Load and preprocess all images
  image_data <- append(image_data, cat_data)  # Append image data
  labels <- append(labels, rep(i - 1, length(cat_images)))  # Assign numeric labels (0, 1, 2)
}



#Convert list of images to a single tensor with 3 channels for RGB
image_tensor <- array(unlist(image_data), 
                      dim = c(length(image_data), img_height, img_width, 3))  # Combine all image

#Convert labels to a tensor
label_tensor <- to_categorical(as.numeric(labels))  #One-hot encode labels for CNNs

#split into training and testing datasets
set.seed(101)
train_indices <- sample(1:nrow(image_tensor), size = 0.8 * nrow(image_tensor))  #80% for training

x_train <- image_tensor[train_indices, , , ]
y_train <- label_tensor[train_indices, ]

x_test <- image_tensor[-train_indices, , , ]
y_test <- label_tensor[-train_indices, ]

#Check tensor shapes. found out just to do this to help me look at data before models
cat("Training data shape:", dim(x_train), "\n")  #Should be (2400, 256, 256, 3)
cat("Training labels shape:", dim(y_train), "\n")  #Should be (2400, 3)
cat("Testing data shape:", dim(x_test), "\n")  #Should be (600, 256, 256, 3)
cat("Testing labels shape:", dim(y_test), "\n")  #Should be (600, 3)

#CNN models


library(keras3)
#simple CNN
model_1 <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = 'relu', input_shape = c(256, 256, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dense(units = 3, activation = 'softmax')

model_1 %>% compile(
  optimizer = 'adam',
  loss = 'categorical_crossentropy',
  metrics = c('accuracy')
)

#larger CNN with more layers
model_2 <- keras_model_sequential() %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu', input_shape = c(256, 256, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 256, activation = 'relu') %>%
  layer_dense(units = 3, activation = 'softmax')

model_2 %>% compile(
  optimizer = 'sgd',
  loss = 'categorical_crossentropy',
  metrics = c('accuracy')
)
#DAN: Maybe this model did poorly because you used sgd. Would have been nice to try all the models with
#the same optimizer and then experiment with different optimizers on the same model, so the two methodological
#explorations are not confounded. 

#combining CNN with some dropout layers
model_3 <- keras_model_sequential() %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu', input_shape = c(256, 256, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(rate = 0.25) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(rate = 0.5) %>%
  layer_flatten() %>%
  layer_dense(units = 512, activation = 'relu') %>%
  layer_dense(units = 3, activation = 'softmax')

model_3 %>% compile(
  optimizer = 'rmsprop',
  loss = 'categorical_crossentropy',
  metrics = c('accuracy')
)


history_1 <- model_1 %>% fit(
  x = x_train, y = y_train,
  epochs = 10,
  batch_size = 32,
  validation_split = 0.2
)
#DAN: Pretty wild that this simple model does so well. However, the model may not be as simple
#as it appears. It's got over 66M params because of the huge dense layers!

history_2 <- model_2 %>% fit(
  x = x_train, y = y_train,
  epochs = 10,
  batch_size = 32,
  validation_split = 0.2
)

history_3 <- model_3 %>% fit(
  x = x_train, y = y_train,
  epochs = 10,
  batch_size = 32,
  validation_split = 0.2
)


test_eval_1 <- model_1 %>% evaluate(x_test, y_test)
test_eval_2 <- model_2 %>% evaluate(x_test, y_test)
test_eval_3 <- model_3 %>% evaluate(x_test, y_test)

cat("Model 1 Test Accuracy:", test_eval_1$accuracy, "\n")
cat("Model 2 Test Accuracy:", test_eval_2$accuracy, "\n")
cat("Model 3 Test Accuracy:", test_eval_3$accuracy, "\n")

#results directory definition
results_dir <- file.path(getwd(), "..", "results")  # Ensure it's relative to the code directory
if (!dir.exists(results_dir)) dir.create(results_dir)
#DAN: Thanks for using relative addressing!

plot_accuracy <- function(history, model_name) {

  history_df <- as.data.frame(history$metrics)
  history_df$epoch <- 1:nrow(history_df)
  

  p <- ggplot(history_df, aes(x = epoch)) +
    geom_line(aes(y = accuracy, color = "Train Accuracy"), size = 1.2) +
    geom_line(aes(y = val_accuracy, color = "Validation Accuracy"), size = 1.2, linetype = "dashed") +
    labs(
      title = paste("Accuracy Over Epochs -", model_name),
      x = "Epoch",
      y = "Accuracy",
      color = "Dataset"
    ) +
    scale_color_manual(values = c("Train Accuracy" = "blue", "Validation Accuracy" = "red")) +
    theme_classic(base_size = 14) +  
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
      legend.position = "bottom",
      legend.title = element_blank()
    )
  
  return(p)
}

# Generate accuracy plots
plot_1 <- plot_accuracy(history_1, "Model 1")
plot_2 <- plot_accuracy(history_2, "Model 2")
plot_3 <- plot_accuracy(history_3, "Model 3")

# Save the improved plots to the results folder
ggsave(file.path(results_dir, "accuracy_plot_model_1.png"), plot = plot_1, width = 8, height = 6)
ggsave(file.path(results_dir, "accuracy_plot_model_2.png"), plot = plot_2, width = 8, height = 6)
ggsave(file.path(results_dir, "accuracy_plot_model_3.png"), plot = plot_3, width = 8, height = 6)

cat("Plots saved to the results folder.\n")
