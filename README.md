# MNIST Digit Recognition with Support Vector Machines

![MNIST Digits](mnist_digits.png)

This repository contains a Python-based project for digit recognition using Support Vector Machines (SVM) on the MNIST dataset. The MNIST dataset is a well-known collection of 28x28 grayscale images of handwritten digits, commonly used for training and evaluating machine learning models.

## Project Overview

The primary objective of this project is to build and fine-tune a Support Vector Machine classifier to recognize handwritten digits from the MNIST dataset. The project includes the following key components:

- **Data Preprocessing**: Data preprocessing is essential to prepare the raw image data for training. This includes reshaping the images into 1D arrays and scaling the pixel values to a standardized range.

- **SVM Model**: We employ a Support Vector Machine classifier, a powerful supervised learning algorithm suitable for both binary and multiclass classification tasks. The SVM is fine-tuned to achieve the best performance.

- **Hyperparameter Optimization**: The project includes a hyperparameter optimization process to find the optimal settings for the SVM model, such as the 'C' parameter, kernel choice, and more.

- **Cross-Validation**: Cross-validation is used to assess the model's performance and evaluate its ability to generalize to unseen data.

- **Accuracy Evaluation**: The project provides a mechanism to evaluate the model's accuracy on both the training and test datasets. Additionally, there's a cross-validation accuracy assessment.

## Getting Started

To get started with this project, follow these steps:

1. Clone the repository to your local machine.
2. Install the required dependencies, which are listed in the project's `requirements.txt`.
3. Run the Jupyter notebooks provided in the repository to explore the code and execute the different stages of the project.

## Usage

You can use this project as a starting point for building a digit recognition system using Support Vector Machines. Feel free to adapt the code, fine-tune the model, and experiment with different hyperparameters to achieve the best performance for your specific use case.

## Acknowledgments

The project is built on the foundation of several open-source libraries and leverages the MNIST dataset. We acknowledge the contributions of the open-source community and the creators of the MNIST dataset for making this project possible.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
