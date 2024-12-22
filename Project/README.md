DAN:

General comments and thinking about what next:

This is an impressive piece of work, both because of what you have managed to accomplish and because
the models achieved impressively high accuracy. If you asked me a couple weeks ago what accuracy I 
guessed simple models of the type you have done might achieve on this problem, I might have said 80%,
with higher accuracies only coming from, say, tranfer learning, or bigger models, or more optimization
efforts. But you got great results with relatively limple models!

I think this already has the kernel of being 1) a modest publication, and 2) a real boon to the people
at UVA doing the barrier island monitoring project. I think you should consider how you want this to
go. Do you just want to hand it off at some point soon to John Porter and be done with it? Do you want 
to hand it off to some student at UVA looking to finish it and make it a workable product they will use
at UVA? Do you want to pursue a publication in addition to one of those options? I see no reason why this
couldn't be published - it would be a solid but not earth-moving publication. Since you are in a mammology
lab, I don't see why this couldn't be a chapter in your thesis, though that's ultimately a conversation
for you to have with your supervisor and your committee. Deciding what you want will help you decide next 
steps.

Regardless what you choose, I think you should send this report to John Porter to let him know where
the project is at and to tell him what you want to do next (based on your above decision). He'd have
the right to be a coauthor on a publication. I think he will be very interested in seeing this report
even if you still intend to do some more work, e.g., run it on a larger computer.

I think if you want to move this closer to a usable product, usable by the monitoring people at UVA,
then you ought to package the model a bit (as you already talk about doing), but I think you also need
to have some functions to do post-processing as follows. Many model predictions will be quite certain.
But some will be uncertain. By attaching uncertainty to your predictions, you can allow the user to
look at the least certain images themselves. This could be set up in many ways and should be done 
after consulting with John Porter. One way would be to just somehow attach the level of certainty to
the prediction, so the user can sort on this and start looking at images starting from the least
certain ones and working until they reach a level of certainty where there are finding they always 
agree with the machine and then they can stop and trust the machine's classifications for images 
past that point. Another way would be to have the user input the level of certainty they want, and 
all images where the classification is more certain than that are just classified, and the remainder
are placed in a separate folder or something for human classification. There are many small choices and
"software engineering lite"-type setup actions that can be taken to facilitate the end user actually 
using the tool you have produced the core of.

Antonios Mamalakis is a UVA professor of data science and environmental science. He knows John Porter.
He is an expert in deep learning, and is also teaching a course on deep learning this spring. He
may have students (either in his lab taking his class) who would be interested in taking this project
over if you just want to hand it off. That person would presumably do further optimizations and
would turn this into a product that the researchers there use. That's only IF you want to hand it off.
Obviously if you want to do this work yourself, you can, since you started this! But if you want to 
hand it off, and John does not want to do it himself (or cannot) you can talk to John and maybe 
Antonios about finding a student there who is interested. I can help make those connections if
you want to make them. 

If you want to make this a useable product, you'll have to work with John on exactly what categories
to use. As you've noticed, there is a long tail of weird stuff. You may want an "other" category
where you just lump all that. Your code and workflow are really quite clean, so I don't think you
necessarily have miles to go to make this a useable product, but there will have to be a step of 
consulting with the people who actually would use it. 

As far as getting the code running on a bigger computer, you can spend hours or days trying to figure
that kind of stuff out when someone can sit down with you for an hour and make it all clear. I suggest
you ask either Luis Madrigal Roca or my postdoc, Angel Robles, if they would sit down with you for a 
defined period of time (e.g., an hour) and help you set up all the environments and whatever else you
need to run this on the cluster. Both of those guys have fought that battle already and can facilitate
it for you while also teaching you how to do it. That's worth an acknowledgement in the final paper 
(if you decide you want to make this into a paper).


More specific comments:

Why did you use different optimizers for different models? That may have complicated the model
comparison. But then again perhaps you tried different optimizers outside the script you have
provided and these are the ones that worked best and for some reason it was different optimizers
for different models? In particular, model 2, which had the worst performance, you used the sgd 
optimizer, and it's not typically considered a good default optimizer (adam and some others 
are considered to typically be better). Like wise for model 3 you used rmsprop, which is considered
to be solid, but it seems people think adam outperforms rmsprop more often than the reverse.

It's a bit of an odd choice to use a transfer learning model (without the last layer) and then
attach to the end an additional several layers. More typical would be just to add the minimum
number of additional layers to get to an output of the format you need. E.g., if the 
transfer learning model returns 2048 filters of 8 by 8 size, you might use a global max pooling
layer to convert these into just 2048 numbers, and then a dense layer with softmax and X outputs
where X is the number of categories. I suggest you consider trying that, or something like it.
The idea is to just use the transfer model, basically. You could try a few more different transfer
learning models if you want - maybe one other than MibileNetV2 works better.


Overall, and grade:
Great work! Grade: 30/30
I'd love it if you keep me updated on the outcomes here. E.g., if you write a publication, please
send it to me and I will make students read it in later iterations of this course! If it becomes a 
product that UVA uses, let me know, and point me to any web presence, and I'll also tell that fact 
to future students of this course so they see what is possible.


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

