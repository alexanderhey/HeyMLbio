#lbraries to load
library(keras3)


project_dir <- getwd()  # project root
image_dir <- file.path(project_dir, "data", "Animals")

categories <- c("cats", "dogs", "snakes")  
img_height <- 256
img_width <- 256

#Function to load, resize, and preprocess RGB images
load_and_preprocess_image <- function(img_path) {
  img <- load.image(img_path)  
  img <- resize(img, img_width, img_height)  #Resize to fixed dimensions. might be useful if working with another dataset, I'm mostly thinking of my final project
  img_array <- as.array(img)  #Convert to array
  img_array <- array(img_array, dim = c(img_height, img_width, 3))  #Add channel dimension for RGB
  return(img_array)
}


#initialize arrays for storing image data and labels
image_data <- list()
labels <- c()

#process images
for (i in seq_along(categories)) { #seq_along in case you have more than a few categories
  cat <- categories[i]
  cat_images <- list.files(path = file.path(image_dir, cat), full.names = TRUE)
  cat_data <- lapply(cat_images, load_and_preprocess_image)  #Load and preprocess all images
  image_data <- append(image_data, cat_data)  #Append image data
  labels <- append(labels, rep(i - 1, length(cat_images)))  #Assign numeric labels (0, 1, 2)
}



#Convert the list of images to a single tensor with 3 channels for RGB
image_tensor <- array(unlist(image_data), 
                      dim = c(length(image_data), img_height, img_width, 3))  # Combine all images

#Convert labels to a tensor for loss categorical crossentropy. to_categorical in keras3. Ex. Label 0 becomes [1, 0, 0]
label_tensor <- to_categorical(as.numeric(labels))  

#Split into training and testing datasets
set.seed(101)
train_indices <- sample(1:nrow(image_tensor), size = 0.8 * nrow(image_tensor))  # 80% for training

x_train <- image_tensor[train_indices, , , ]
y_train <- label_tensor[train_indices, ]

x_test <- image_tensor[-train_indices, , , ]
y_test <- label_tensor[-train_indices, ]

# Check tensor shapes. I like keeping this here for the user
cat("Training data shape:", dim(x_train), "\n")  # Should be (2400, 256, 256, 3)
cat("Training labels shape:", dim(y_train), "\n")  # Should be (2400, 3)
cat("Testing data shape:", dim(x_test), "\n")  # Should be (600, 256, 256, 3)
cat("Testing labels shape:", dim(y_test), "\n")  # Should be (600, 3)


#Load MobileNetV2 pre-trained model without top layers
base_model <- application_mobilenet_v2(
  include_top = FALSE,          #Exclude  original classifier
  weights = "imagenet",         #Use pre-trained ImageNet weights
  input_shape = c(256, 256, 3)  #Match input shape from before 
)

#freeze base model with freeze_weight. nifty function! 
freeze_weights(base_model)

#I want the model to be the same model that performed well in the last day to see if we can get it better. I don't know how much better it will get, as that model was pretty awesome
transfer_model <- keras_model_sequential() %>%
  base_model %>%  
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = 'relu') %>%  
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%                                
  layer_flatten() %>%                                                          
  layer_dense(units = 128, activation = 'relu') %>%                            
  layer_dense(units = 3, activation = 'softmax')        


transfer_model %>% compile(
  optimizer = 'adam',
  loss = 'categorical_crossentropy',
  metrics = c('accuracy')
)


#train with my variables loaded in from Day-3.R

history_transfer <- transfer_model %>% fit(
  x = x_train, 
  y = y_train, 
  epochs = 10, 
  batch_size = 32, 
  validation_data = list(x_test, y_test)
)

test_eval <- transfer_model %>% evaluate(x_test, y_test)
test_eval2<- model_1 %>% evaluate(x_test, y_test)

transfer_accuracy <- test_eval$accuracy
previous_accuracy <- test_eval2$accuracy

cat("Transfer Learning Model Accuracy:", transfer_accuracy, "\n")
cat("Previous Best Model Accuracy:", previous_accuracy, "\n")


# results directory
results_dir <- file.path(getwd(), "results")  # Set results folder relative to the working directory
if (!dir.exists(results_dir)) dir.create(results_dir)  # Create folder if it doesn't exist

# Function to plot accuracy with improved readability
library(ggplot2)
plot_accuracy <- function(history, model_name) {
  #Convert history metrics to a data frame
  history_df <- as.data.frame(history$metrics)
  history_df$epoch <- 1:nrow(history_df)
  
  #Create accuracy plot with a white background and improved readability
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
    theme_classic(base_size = 14) +  #White background and clean layout
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
      legend.position = "bottom",
      legend.title = element_blank()
    )
  
  return(p)
}

#Generate accuracy plot for transfer model
plot_transfer <- plot_accuracy(history_transfer, "Transfer Model")

#Save the plot to the results folder
output_file <- file.path(results_dir, "transfer_model_accuracy.png")  #Path to results folder
ggsave(output_file, plot = plot_transfer, width = 8, height = 6)  #Save plot to results folder

#Print confirmation message
cat("Saved transfer model accuracy plot to:", output_file, "\n")
