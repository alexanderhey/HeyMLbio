#Image Classification of MouseCam Animals- Convolutional Neural Network model selection script
#make sure you process images accordingly in the process_images.R script 

#simple CNN
model_1 <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = 'relu', input_shape = c(256, 256, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dense(units = 13, activation = 'softmax')  # Update to 13 units


model_1 %>% compile(
  optimizer = 'adam',
  loss = 'categorical_crossentropy',
  metrics = c('accuracy')
)

history_1 <- model_1 %>% fit(
  x = x_train, y = y_train,
  epochs = 10,
  batch_size = 32,
  validation_split = 0.2
)

# Larger CNN with more layers
model_2 <- keras_model_sequential() %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu', input_shape = c(256, 256, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 256, activation = 'relu') %>%
  layer_dense(units = 13, activation = 'softmax')  # Changed to 13

model_2 %>% compile(
  optimizer = 'sgd',
  loss = 'categorical_crossentropy',
  metrics = c('accuracy')
)

# Combining CNN with dropout layers
model_3 <- keras_model_sequential() %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu', input_shape = c(256, 256, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(rate = 0.25) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(rate = 0.5) %>%
  layer_flatten() %>%
  layer_dense(units = 512, activation = 'relu') %>%
  layer_dense(units = 13, activation = 'softmax')  # Changed to 13

model_3 %>% compile(
  optimizer = 'rmsprop',
  loss = 'categorical_crossentropy',
  metrics = c('accuracy')
)

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

# Evaluation
test_eval_1 <- model_1 %>% evaluate(x_test, y_test)
test_eval_2 <- model_2 %>% evaluate(x_test, y_test)
test_eval_3 <- model_3 %>% evaluate(x_test, y_test)

cat("Model 1 Test Accuracy:", test_eval_1$accuracy, "\n")
cat("Model 2 Test Accuracy:", test_eval_2$accuracy, "\n")
cat("Model 3 Test Accuracy:", test_eval_3$accuracy, "\n")


#results directory definition
results_dir <- file.path(getwd(), "..", "results")  # Ensure it's relative to the code directory
if (!dir.exists(results_dir)) dir.create(results_dir)


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


# Function to save images with corrected handling of RGB tensor
save_image_magick <- function(index, type, results_dir) {
  # Extract the image and rescale to [0, 255]
  img_array <- x_test[index, , , ] * 255
  
  # Reshape and convert the tensor to a format compatible with magick
  img_rgb <- array(as.integer(img_array), dim = c(256, 256, 3))  # Ensure integer RGB values
  img_rgb <- aperm(img_rgb, c(2, 1, 3))  # Transpose axes for magick compatibility
  
  #Convert to magick image (found online)
  img <- image_read(as.raster(img_rgb / 255))  # Normalize to [0, 1] for magick
  
  #Annotate image. I like adding this
  predicted_label <- categories[predicted_classes_model_3[index] + 1]
  actual_label <- categories[actual_classes[index] + 1]
  annotated_img <- image_annotate(
    img,
    text = paste(type, "\nPredicted:", predicted_label, "\nActual:", actual_label),
    color = "white", size = 14, location = "+10+10"
  )
  
  # Save the image
  filename <- file.path(results_dir, paste0(type, "_image_", index, ".png"))
  image_write(annotated_img, path = filename, format = "png")
  message("Saved ", type, " image as: ", filename)
}

# Save correctly classified images
cat("Saving correctly classified images...\n")
for (i in seq_len(min(5, length(correctly_classified_indices_model_3)))) {
  save_image_magick(correctly_classified_indices_model_3[i], "Model3_correctly_classified", results_dir)
}

# Save misclassified images
cat("Saving misclassified images...\n")
for (i in seq_len(min(5, length(misclassified_indices_model_3)))) {
  save_image_magick(misclassified_indices_model_3[i], "Model3_misclassified", results_dir)
}


# Predict classes using Model 3
predicted_classes_model_3 <- model_3 %>% predict(x_test) %>% apply(1, which.max) - 1  # Adjust indexing
actual_classes <- apply(y_test, 1, which.max) - 1

#Classification statz
correctly_classified_indices_model_3 <- which(predicted_classes_model_3 == actual_classes)
misclassified_indices_model_3 <- which(predicted_classes_model_3 != actual_classes)

cat("Model 3 - Number of correctly classified images:", length(correctly_classified_indices_model_3), "\n")
cat("Model 3 - Number of misclassified images:", length(misclassified_indices_model_3), "\n")


# Visualize classification statistics
classification_stats_model_3 <- data.frame(
  Class = c("Correctly Classified", "Misclassified"),
  Count = c(length(correctly_classified_indices_model_3), length(misclassified_indices_model_3))
)

#Bar plot for classification results
plot_stats_model_3 <- ggplot(classification_stats_model_3, aes(x = Class, y = Count, fill = Class)) +
  geom_bar(stat = "identity", color = "black") +
  theme_minimal() +
  labs(
    title = "Classification Statistics - Model 3",
    x = "Classification Type",
    y = "Number of Images"
  ) +
  scale_fill_manual(values = c("Correctly Classified" = "green", "Misclassified" = "red"))

# Save the classification statistics plot
plot_filename_model_3 <- file.path(results_dir, "classification_statistics_model_3.png")
ggsave(plot_filename_model_3, plot = plot_stats_model_3, width = 8, height = 6)
cat("Saved classification statistics plot for Model 3 as:", plot_filename_model_3, "\n")
