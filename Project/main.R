# ---------------------------
# Main Script for Image Pipeline of Processing Image Data
# ---------------------------

#Load the required libraries
library(readxl)
library(httr)
library(dplyr)
library(imagefx)
library(magick)
library(keras)
library(imager)
library(jpeg)
library(ggplot2)
library(keras3)
library(readxl)
library(keras3)


# ---------------------------
# Set Up Project Directories
# ---------------------------
project_dir <- normalizePath(".")  #Root directory (project)
code_dir <- file.path(project_dir, "code")      #Code directory. Make sure all the scripts (.R) are here
data_dir <- file.path(project_dir, "data")      #Data directory
results_dir <- file.path(project_dir, "results")  #Results directory

# ---------------------------
# Run Scripts Sequentially
# ---------------------------
cat("Starting preprocessing images...\n")
source(file.path(code_dir, "preprossess_images.R"))
cat("Preprocessing completed.\n")

cat("Starting model selection...\n")
source(file.path(code_dir, "model_selection.R"))
cat("Model selection completed.\n")

cat("Starting transfer learning...\n")
source(file.path(code_dir, "transfer_learning.R"))
cat("Transfer learning completed.\n")

