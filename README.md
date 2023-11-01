# MNIST Digit Recognition

This repository contains a Python-based project for digit recognition using on the MNIST dataset. The MNIST dataset is a well-known collection of 28x28 grayscale images of handwritten digits, commonly used for training and evaluating machine learning models.

# Models 

## - Support Vector Machines (SVM)

- We started by implementing a Support Vector Machine (SVM), a well-established machine learning algorithm, known for its effectiveness in classification tasks. The SVM model was fine-tuned and optimized using grid search and cross-validation techniques.
- **Data Preprocessing**: Data preprocessing is essential to prepare the raw image data for training. This includes reshaping the images into 1D arrays and scaling the pixel values to a standardized range.

- **SVM Model**: We employ a Support Vector Machine classifier, a powerful supervised learning algorithm suitable for both binary and multiclass classification tasks. The SVM is fine-tuned to achieve the best performance.

- **Hyperparameter Optimization**: The project includes a hyperparameter optimization process to find the optimal settings for the SVM model, such as the 'C' parameter, kernel choice, and more.

- **Cross-Validation**: Cross-validation is used to assess the model's performance and evaluate its ability to generalize to unseen data.

- **Accuracy Evaluation**: The project provides a mechanism to evaluate the model's accuracy on both the training and test datasets. Additionally, there's a cross-validation accuracy assessment.

  
## - Convolutional Neural Network (CNN)

- In addition to the SVM, we introduced a Convolutional Neural Network (CNN) for digit recognition. CNNs are deep learning models tailored for image classification tasks. Our CNN architecture includes convolutional layers, max-pooling layers, and fully connected layers, allowing us to learn complex features directly from pixel data. The model was trained on the MNIST dataset, leveraging spatial relationships within the images to achieve high accuracy in recognizing handwritten digits.

## Project Overview

The primary objective of this project is to build and fine-tune a different classifier to recognize handwritten digits from the MNIST dataset.

## Required Packages

These packages are needed to be installed to properly run this project:

- **scikit-learn**
- **numpy**
- **matplotlib**
- **tensorflow**
- **pandas** -> optional for file printing as CSV

