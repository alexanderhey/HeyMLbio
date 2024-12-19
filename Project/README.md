# Image Classification of MouseCam Animals

## Project Overview
This project focuses on developing a robust image classification pipeline for MouseCam images captured by Dr. John Porter at the University of Virginia. The primary goal was to create a neural network capable of classifying images into various animal categories. Due to unbalanced classifications and resource constraints, preprocessing techniques were employed to reduce image size and balance the dataset. The project was designed to allow scalability for future analysis on more powerful hardware.

### Key Features
- Preprocessing images by cropping and resizing.
- Balancing the dataset by limiting the number of images per category.
- Building three CNN architectures of increasing complexity.
- Implementing transfer learning using MobileNetV2.
- Visualizing training performance and evaluating model accuracy.

---

## Project Structure

```
project_root/
├── code/
│   ├── preprocess_images.R       # Image preprocessing script
│   ├── model_selection.R         # Building and training CNNs
│   ├── transfer_learning.R       # Transfer learning using MobileNetV2
├── data/
│   ├── images_data.xlsx          # Metadata for image dataset
│   ├── downloaded_images/        # Folder for downloaded images
│   ├── resized_images/           # Folder for resized images
├── results/
│   ├── accuracy_plot_model_1.png # Accuracy plot for simple CNN
│   ├── accuracy_plot_model_2.png # Accuracy plot for larger CNN
│   ├── accuracy_plot_model_3.png # Accuracy plot for CNN with dropout
│   ├── accuracy_plot_transfer_model.png # Accuracy plot for transfer model
│   ├── correctly_classified_*.png # Correctly classified image examples
│   ├── misclassified_*.png       # Misclassified image examples
```

---

## Data Preprocessing

### Steps
1. **Downloading and filtering images:**
   - Removed bottom 100 pixels to eliminate camera metadata.
   - Cropped 150 pixels from each side to exclude non-essential parts.
2. **Resizing images:** Resized to 256x256 pixels for uniformity.
3. **Splitting dataset:** Divided into training (80%) and testing (20%) sets.

### Dataset Summary
- **Training set:** 80% of the images.
- **Testing set:** 20% of the images.

---

## Model Training and Selection

### Models

#### 1. Simple CNN
- Single convolutional layer, max-pooling, and two dense layers.
- Test Accuracy: 97.98%

#### 2. Larger CNN
- Additional convolutional layer and increased filters.
- Test Accuracy: 95.65%

#### 3. CNN with Dropout Layers
- Dropout layers added to reduce overfitting.
- Test Accuracy: 98.29%

### Accuracy Plots
Accuracy plots for each model are available in the `results/` directory.

---

## Transfer Learning
Transfer learning was applied using the MobileNetV2 architecture to evaluate its performance compared to custom CNN models. While it achieved a test accuracy of 96.42%, Model 3 remained the best-performing model.

---

## Results

### Classification Statistics
#### Model 3 (Best Performing)
- Correctly classified images: 632
- Misclassified images: 11

#### Transfer Learning Model
- Correctly classified images: 620
- Misclassified images: 23

### Visualization Examples
Examples of correctly and misclassified images for all models are saved in the `results/` directory.

---

## Requirements

### Software
- R (v4.0 or later)
- Required R libraries:
  - `keras3`
  - `ggplot2`
  - `magick`
  - `imager`
  - `readxl`
  - `dplyr`
  - `httr`
  - `keras`
  - `jpeg`
  - `imagefx`
  

### Hardware
- GPU-enabled machine for faster training (recommended).I ran this code on a Dell XPS 15 7590.

---

## How to Run (You can do this from the main.R script)

1. **Set Up Environment:**
   - Install all required R libraries.

2. **Prepare Data:**
   - Place `images_data.xlsx` in the `data/` directory.
   - Run `preprocess_images.R` to download and preprocess images.

3. **Train Models:**
   - Run `model_selection.R` to train CNN models and generate evaluation metrics.

4. **Apply Transfer Learning:**
   - Run `transfer_learning.R` to fine-tune MobileNetV2 and evaluate its performance.

5. **View Results:**
   - Check the `results/` directory for evaluation plots and examples of classified images.

---

## Conclusion
Model 3 demonstrated the best overall performance with a test accuracy of 98.29%, highlighting the effectiveness of dropout layers in reducing overfitting.

---

