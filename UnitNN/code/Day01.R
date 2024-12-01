#so I lost all my code for some reason when trying to compile and organize my data into folders before submission. I overwrote the original file and lost it, and I should have been making commits to github to get away from this issue.
#lesson learned. Below is basically the same code form day 2 and it runs how the initial code did, it just saves a slightly different plot and doesn't have my initial comments about my frustration with this dataset, haha




#Image directories
project_dir <- getwd()  # Get the current working directory (project root)
image_dir <- file.path(project_dir, "data", "Animals")

#Folders/categories
categories <- c("cats", "dogs", "snakes")

img_height <- 256
img_width <- 256

#Function to load and resize images
load_and_resize_image <- function(img_path) {
  img <- load.image(img_path)  # Load image
  img <- resize(img, img_width, img_height)  #Resize image
  img <- grayscale(img)  #Convert to grayscale
  as.numeric(img)  #Convert image to a numeric vector
}

#Initialize lists for storing image data and labels
image_data <- list()
labels <- c()

#Loop through each category and process up to 100 images
for (cat in categories) {
  cat_images <- list.files(path = file.path(image_dir, cat), full.names = TRUE)
  
  #Limit to 100 images per category
  cat_images <- head(cat_images, 100)
  
  #Load and process images
  cat_data <- lapply(cat_images, load_and_resize_image)  #Load and resize all images
  image_data <- append(image_data, cat_data)  #Append image data
  labels <- append(labels, rep(cat, length(cat_images)))  #Create corresponding labels
}

#Convert image data and labels to a dataframe
image_df <- as.data.frame(do.call(rbind, image_data))  # Combine all image vectors into one dataframe
image_df$label <- as.factor(labels)  # Add labels as a factor

#Check structure of the dataframe
str(image_df)

#Load thinned dataset
data <- image_df

#Extract predictors (image data) and response variable (labels)
x <- as.matrix(data[, -ncol(data)])  #All columns except the last
y <- as.numeric(as.factor(data$label)) - 1  #Convert labels to numeric starting at 0

#Normalize the predictors (pixel values range from 0 to 255, scale to 0-1)
x <- x / 255

#Split into training and testing sets
set.seed(123)
train_indices <- sample(1:nrow(x), size = 0.8 * nrow(x))  # 80% for training
x_train <- x[train_indices, ]
y_train <- y[train_indices]
x_test <- x[-train_indices, ]
y_test <- y[-train_indices]

#Flatten the 4D data into 2D
x_train_flat <- array_reshape(x_train, c(dim(x_train)[1], img_height * img_width))
x_test_flat <- array_reshape(x_test, c(dim(x_test)[1], img_height * img_width))

#Define the fully connected model. For some reason, this was the only way to input my parameters for Day 1 and 2. I think it has to do with how I preprocessed data. I talked about it a lot in my intitial code, but I can't remember the errors I would get before. I think it had something to do with TensorFlow issues
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


results_dir <- file.path(getwd(), "results")  
if (!dir.exists(results_dir)) dir.create(results_dir)

save_accuracy_plot <- function(history, filename) {
  #Extract accuracy and validation accuracy
  train_acc <- history$metrics$accuracy
  val_acc <- history$metrics$val_accuracy
  epochs <- seq_along(train_acc)
  
  plot(epochs, train_acc, type = "l", col = "blue", ylim = c(0, 1),
       xlab = "Epoch", ylab = "Accuracy", main = paste("Accuracy Plot:", filename))
  lines(epochs, val_acc, col = "red")
  legend("bottomright", legend = c("Training", "Validation"),
         col = c("blue", "red"), lty = 1)
  
  #save
  full_filename <- file.path(results_dir, paste0(filename, "_accuracy_plot.png"))  # EDIT HERE
  dev.copy(png, full_filename)
  dev.off()
  message("Saved plot as: ", full_filename)
}

save_accuracy_plot(history, "training_plots")

