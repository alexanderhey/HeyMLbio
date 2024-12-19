#Image Classification of MouseCam Animals- Transfer Learning
#This script can be run to test different transfer learning models in Keras. I used my best model from the model_selection.R script combined with a transfer learning model.

#Load MobileNetV2 pre-trained model without top layers
base_model <- application_mobilenet_v2(
  include_top = FALSE,          # Exclude original classifier
  weights = "imagenet",         # Use pre-trained ImageNet weights
  input_shape = c(256, 256, 3)  # Match input shape
)

#Freeze the base model weights
freeze_weights(base_model)

#Define new model on top of the frozen base model
inputs <- layer_input(shape = c(256, 256, 3))  # Input layer

#Pass inputs through the base model
x <- base_model(inputs, training = FALSE)  # Frozen base model

# Add custom layers
x <- layer_conv_2d(x, filters = 64, kernel_size = c(3, 3), activation = 'relu', padding = "same") %>%
  layer_dropout(rate = 0.25) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = 'relu', padding = "same") %>%
  layer_dropout(rate = 0.5) %>%
  layer_global_average_pooling_2d() %>%   # Replace MaxPooling with GlobalAveragePooling
  layer_dense(units = 512, activation = 'relu') %>%
  layer_dense(units = 13, activation = 'softmax')  # Final output layer for 13 classes

#Define the complete transfer model
transfer_model <- keras_model(inputs = inputs, outputs = x)

#Compile the model
transfer_model %>% compile(
  optimizer = optimizer_adam(),  # Adam optimizer
  loss = 'categorical_crossentropy',
  metrics = c('accuracy')
)

#Train the model
history_transfer <- transfer_model %>% fit(
  x = x_train, 
  y = y_train, 
  epochs = 10, 
  batch_size = 32, 
  validation_data = list(x_test, y_test)
)

#Evaluate the model
test_eval <- transfer_model %>% evaluate(x_test, y_test)
cat("Transfer Learning Model Test Accuracy:", test_eval$accuracy, "\n")

# Function to plot accuracy
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

#Generate and save accuracy plot for transfer model
plot_transfer <- plot_accuracy(history_transfer, "Transfer Learning Model")
ggsave(file.path(results_dir, "accuracy_plot_transfer_model.png"), plot = plot_transfer, width = 8, height = 6)
cat("Saved transfer learning accuracy plot to:", results_dir, "\n")

#Predict
predicted_classes_transfer <- transfer_model %>% predict(x_test) %>% apply(1, which.max) - 1
actual_classes <- apply(y_test, 1, which.max) - 1

#Classification statistics
correctly_classified_indices <- which(predicted_classes_transfer == actual_classes)
misclassified_indices <- which(predicted_classes_transfer != actual_classes)

cat("Transfer Model - Number of correctly classified images:", length(correctly_classified_indices), "\n")
cat("Transfer Model - Number of misclassified images:", length(misclassified_indices), "\n")

#Function to save correctly and misclassified images
save_image_magick <- function(index, type, results_dir) {
  #Extract and rescale image
  img_array <- x_test[index, , , ] * 255
  
  #dimensions and transpose for plotting
  img_rgb <- array(as.integer(img_array), dim = c(256, 256, 3))
  img_rgb <- aperm(img_rgb, c(2, 1, 3))  # Transpose
  
  #Convert to magick object (found online)
  img <- image_read(as.raster(img_rgb / 255))  # Normalize for magick
  
  #Annotate image. I like adding this 
  predicted_label <- categories[predicted_classes_transfer[index] + 1]
  actual_label <- categories[actual_classes[index] + 1]
  annotated_img <- image_annotate(
    img,
    text = paste(type, "\nPredicted:", predicted_label, "\nActual:", actual_label),
    color = "white", size = 14, location = "+10+10"
  )
  
  #Save image
  filename <- file.path(results_dir, paste0(type, "_image_", index, ".png"))
  image_write(annotated_img, path = filename, format = "png")
  message("Saved ", type, " image as: ", filename)
}

#Save examples of correctly classified images
cat("Saving correctly classified images...\n")
for (i in seq_len(min(5, length(correctly_classified_indices)))) {
  save_image_magick(correctly_classified_indices[i], "correctly_classified_transfer_model", results_dir)
}

#Save examples of misclassified images
cat("Saving misclassified images...\n")
for (i in seq_len(min(5, length(misclassified_indices)))) {
  save_image_magick(misclassified_indices[i], "misclassified_transfer_model", results_dir)
}

