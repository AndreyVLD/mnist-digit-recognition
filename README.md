# MNIST Digit Recognition with Support Vector Machines

This repository contains a Python-based project for digit recognition using Support Vector Machines (SVM) on the MNIST dataset. The MNIST dataset is a well-known collection of 28x28 grayscale images of handwritten digits, commonly used for training and evaluating machine learning models.

## Project Overview

The primary objective of this project is to build and fine-tune a Support Vector Machine classifier to recognize handwritten digits from the MNIST dataset. The project includes the following key components:

- **Data Preprocessing**: Data preprocessing is essential to prepare the raw image data for training. This includes reshaping the images into 1D arrays and scaling the pixel values to a standardized range.

- **SVM Model**: We employ a Support Vector Machine classifier, a powerful supervised learning algorithm suitable for both binary and multiclass classification tasks. The SVM is fine-tuned to achieve the best performance.

- **Hyperparameter Optimization**: The project includes a hyperparameter optimization process to find the optimal settings for the SVM model, such as the 'C' parameter, kernel choice, and more.

- **Cross-Validation**: Cross-validation is used to assess the model's performance and evaluate its ability to generalize to unseen data.

- **Accuracy Evaluation**: The project provides a mechanism to evaluate the model's accuracy on both the training and test datasets. Additionally, there's a cross-validation accuracy assessment.
