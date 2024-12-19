#Image Classification of MouseCam Animals- processing image data
#This resizes, crops, and arranges your data into the proper space for Neural Networks using Keras


#Reading in File
data_dir <- file.path(project_dir, "data")  # Path to `data` directory
file_path <- file.path(data_dir, "images_data.xlsx")  # Path to Excel file
data <- read_excel(file_path)
download_dir <- file.path(data_dir, "downloaded_images")  # Path for downloaded images
resized_dir <- file.path(data_dir, "resized_images")  # Path for resized images


#Filter images captured after 2020 and with a non-empty Species column
filtered_data <- data %>%
  filter(!is.na(Species), as.Date(DateTimeOriginal) > as.Date("2020-12-31"))


if (!dir.exists(download_dir)) {
  dir.create(download_dir)
}

#Function to download and save images locally
download_images <- function(url, save_path) {
  tryCatch({
    download.file(url, save_path, mode = "wb")
    message("Downloaded: ", url)
  }, error = function(e) {
    message("Failed to download: ", url)
  })
}

#Iterate over filtered data to download images
for (i in seq_len(nrow(filtered_data))) {
  file_url <- filtered_data$FileURL[i]
  species <- filtered_data$Species[i]
  save_subdir <- file.path(download_dir, species)
  
  #Create species subdirectory (if it doesn't exist)
  if (!dir.exists(save_subdir)) {
    dir.create(save_subdir)
  }
  
  #Set save path for the image
  file_name <- basename(file_url)
  save_path <- file.path(save_subdir, file_name)
  
  #Download the image
  download_images(file_url, save_path)
}


#Image Manipulation

image_dir <- download_dir

#Function: crop the bottom 150 pixels of an image and replace the original (risky)
crop_and_replace <- function(image_path, crop_pixels) {
  #Load the image
  img <- image_read(image_path)
  
  #Get image dimensions
  img_info <- image_info(img)
  img_width <- img_info$width
  img_height <- img_info$height
  
  #Define cropping geometry
  crop_geometry <- sprintf("%dx%d+0+0", img_width, img_height - crop_pixels)
  
  #Crop the image
  img_cropped <- image_crop(img, crop_geometry)
  
  #Overwrite the original image with the cropped one
  #I did this to save space on my poor laptop, likely best to save to a dif directory instead of override
  image_write(img_cropped, path = image_path)
}

#Process all images in the directory (including subdirectories)
all_images <- list.files(path = image_dir, full.names = TRUE, recursive = TRUE)

# Loop through and crop each image
for (img_path in all_images) {
  crop_and_replace(img_path, crop_pixels = 100)
  cat(sprintf("Processed: %s\n", img_path))  # Log progress
}

cat("All images cropped and replaced.\n")


#Function: crop 1000 pixels from each side of an image and replace the original
crop_sides_and_replace <- function(image_path, crop_pixels_side) {
  #Load the image
  img <- image_read(image_path)
  
  #Get image dimensions
  img_info <- image_info(img)
  img_width <- img_info$width
  img_height <- img_info$height
  
  #cropping geometry
  new_width <- img_width - (2 * crop_pixels_side)  #Subtract from both sides
  crop_geometry <- sprintf("%dx%d+%d+0", new_width, img_height, crop_pixels_side)
  
  #Crop the image
  img_cropped <- image_crop(img, crop_geometry)
  
  # Overwrite the original image with the cropped one
  image_write(img_cropped, path = image_path)
}


#Process all images in the directory (including subdirectories)
all_images <- list.files(path = image_dir, full.names = TRUE, recursive = TRUE)

#Loop through and crop each image
for (img_path in all_images) {
  crop_sides_and_replace(img_path, crop_pixels_side = 150)
  cat(sprintf("Processed: %s\n", img_path))  # Log progress
}

cat("All images cropped and replaced.\n")



#Resizing images to save space. reduces the amount of data per images
#Directories
project_dir <- getwd()  # Root directory
image_dir <- file.path(project_dir, "downloaded_images")  
output_dir <- file.path(project_dir, "resized_images")    
if (!dir.exists(output_dir)) dir.create(output_dir)       

#Parameters. Reason: I ran other NN and used these dimensions.
img_height <- 256               
img_width <- 256                

#Function to load and resize an image
resize_image <- function(img_path, output_path) {
  tryCatch({
    # Load it in
    img <- image_read(img_path)
    
    #resize and preserve aspect ratio
    img_resized <- image_resize(img, geometry = sprintf("%dx%d", img_width, img_height))
    
    # Save the resized image
    image_write(img_resized, path = output_path)
    cat("Resized and saved:", output_path, "\n")
  }, error = function(e) {
    cat("Error resizing image:", img_path, "\n", e$message, "\n")
  })
}
categories <- list.dirs(download_dir, full.names = FALSE, recursive = FALSE)

for (cat in categories) {
  cat_dir <- file.path(download_dir, cat)
  cat_output_dir <- file.path(resized_dir, cat)
  if (!dir.exists(cat_output_dir)) dir.create(cat_output_dir)
  cat_images <- list.files(cat_dir, full.names = TRUE, pattern = "(?i)\\.(jpg|png|jpeg)$")
  for (img_path in cat_images) {
    output_path <- file.path(cat_output_dir, basename(img_path))
    resize_image(img_path, output_path)
  }
}


#Processing for NN. I feel like this makes the images more blurry as well
image_dir <- resized_dir  #Directory containing images

#Define the categories. I only chose those with "one" mammal in the image. I also wanted it to be dynamic in case you get more species later, thus this allows you to build a better model in the future with mmore categories
categories <- c("Bird", "Crab", "GlaucomysVolans", "Insect", "MicrotusPennsylvanicus", 
                "MusMusculus", "OrozomysPalustris", "PeromyscusLeucopus", "ProcyonLotor", 
                "RattusNorvegicus", "Snake", "SorexSp", "Sylvilagus")

#Number of images to load per category (max). I did 500 because of my laptop limitations, Increasing if on PC or something powerful like a hpc
numload <- 500

# Initialize lists to dynamically store images and labels
image_list <- list()
label_list <- c()

#Load images from each category
for (i in seq_along(categories)) {
  cat("Loading images for category:", categories[i], "\n")
  
  #List image files
  img_files <- list.files(
    path = paste0(image_dir, categories[i], "/"), 
    full.names = TRUE, 
    pattern = "(?i)\\.jpe?g$"
  )
  
  #Skip empty categories
  if (length(img_files) == 0) {
    cat("No images found for category:", categories[i], "- Skipping.\n")
    next
  }
  
  #Limit to `numload` images or the number available. A helpful output message
  num_to_load <- min(length(img_files), numload)
  cat("Number of images to load:", num_to_load, "\n")
  
  #Load
  for (j in 1:num_to_load) {
    img <- tryCatch({
      load.image(img_files[j])  
    }, error = function(e) {
      cat("Error loading image:", img_files[j], "\n")
      return(NULL)
    })
    
    if (is.null(img)) next
    
    # Resize to 256x256
    img <- resize(img, size_x = 256, size_y = 256)
    img <- as.array(img)
    
    # Ensure RGB channels
    if (length(dim(img)) == 2) {  # Grayscale image. some of those nightime shots might be this 
      img <- array(rep(img, 3), dim = c(256, 256, 3))
    }
    
    # Append the image and label
    image_list <- append(image_list, list(img))
    label_list <- c(label_list, i - 1)  # Numeric label
  }
}

# Convert the list of images to an array and labels to a numeric vector
num_images <- length(image_list)
x <- array(NA, dim = c(num_images, 256, 256, 3))  # Initialize an empty array
for (i in seq_along(image_list)) {
  x[i, , , ] <- image_list[[i]]  # Fill the array
}
y <- as.numeric(label_list)

#Shuffle
set.seed(101)
inds <- sample(1:length(y), size = length(y), replace = FALSE)
x <- x[inds, , , ]
y <- y[inds]

#Convert labels to categorical format
y_categorical <- to_categorical(y)

#split into training and testing datasets
split <- 0.2  #percent for testing
test_size <- round(split * length(y))

x_test <- x[1:test_size, , , ]
x_train <- x[(test_size + 1):length(y), , , ]
y_test <- y_categorical[1:test_size, ]
y_train <- y_categorical[(test_size + 1):length(y), ]

#check tensor shapes
cat("Your data shapes should match the amount of images in your dataset, followed by your dimensions followed by the amount of categories you have!")
cat("Training data shape:", dim(x_train), "\n")
cat("Training labels shape:", dim(y_train), "\n")
cat("Testing data shape:", dim(x_test), "\n")
cat("Testing labels shape:", dim(y_test), "\n")



#plot an image just to make sure it looks good. I got a lot of corruption before I got it right so this is a sanity check
plot_image <- function(image_array, index, labels = NULL, categories = NULL) {
  img <- image_array[index, , , ]
  
  #ensure pixel values are in [0, 1]
  if (min(img) < 0) {
    img <- (img + 1) / 2  #Rescale from [-1, 1] to [0, 1]
  }
  
  #Transpose
  img <- aperm(img, c(2, 1, 3))
  img_cimg <- as.cimg(img)
  
  #Plot the image with label
  if (!is.null(labels) && !is.null(categories)) {
    label <- categories[which.max(labels[index, ]) ]
    plot(img_cimg, main = paste("Category:", label))
  } else {
    plot(img_cimg, main = paste("Image Index:", index))
  }
}

# Test plot an image
test_index <- 120  # Change this to plot other images
plot_image(x_train, test_index, y_train, categories)



#Save preprocessed data 
saveRDS(x_train, file = file.path(data_dir,"x_train_final.rds"))
saveRDS(y_train, file = file.path(data_dir,"y_train_final.rds"))
saveRDS(x_test, file = file.path(data_dir,"x_test_final.rds"))
saveRDS(y_test, file = file.path(data_dir,"y_test_final.rds"))

cat("Data saved successfully.\n")

