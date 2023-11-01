from sklearn.model_selection import train_test_split
from mnist.SVM import SVMClassifier as SVM
from mnist.CNN import CNNClassifier as CNN
import numpy as np
import pandas as pd
from keras.datasets import mnist


def file_printer(prediction, estimated_accuracy, filename):
    result = np.append(estimated_accuracy, prediction)
    pd.DataFrame(result).to_csv(filename, index=False, header=False)


# Load data
# Lab dataset:
mnist_28x28_lab = np.load('data/mnist_28x28_train.npy')
labels_lab = np.load('data/mnist_labels.npy')
mnist_28x28_uk_lab = np.load('data/mnist_28x28_unknown.npy')

# MNIST official dataset:
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Splitting the data for CNN
x_train_lab, x_test_lab, y_train_lab, y_test_lab = train_test_split(mnist_28x28_lab, labels_lab, test_size=0.2,
                                                                    random_state=42)


# SVM
def svm(uses_lab_data=True):
    if uses_lab_data:
        svm_classifier = SVM(mnist_28x28_lab, labels_lab)
    else:
        svm_classifier = SVM(x_train, y_train)
    svc_param_grid = {
        'C': [2.745],
        'kernel': ['rbf'],
        'degree': [0],
        'gamma': ['scale'],
        'random_state': [42]
    }
    print("\n-------------------------------------------------------------")
    print("SVC OUTPUT:")
    print("Best hyperparameters: " + svm_classifier.fine_tune_svm(param_grid=svc_param_grid).__str__())
    print("Accuracy with cross-validation: " + svm_classifier.cross_val_accuracy().__str__())
    print("Overall accuracy with lab data: " + svm_classifier.overall_accuracy(mnist_28x28_lab,
                                                                               labels_lab).__str__())


# CNN
def cnn(uses_lab_data=True):
    # Split data for CNN
    print("\n-------------------------------------------------------------")
    print("CNN OUTPUT:")

    if uses_lab_data:
        X_train = x_train_lab
        Y_train = y_train_lab
        X_test = x_test_lab
        Y_test = y_test_lab
        batch_size = 8
    else:
        X_train = x_train
        Y_train = y_train
        X_test = x_test
        Y_test = y_test
        batch_size = 500

    cnn_classifier = CNN(batch_size)
    cnn_classifier.train(X_train, Y_train, X_test, Y_test, epochs=30)
    accuracy = cnn_classifier.evaluate(X_test, Y_test)
    cnn_classifier.plot_history()

    print("Accuracy: " + accuracy.__str__())
    file_printer(cnn_classifier.predict(mnist_28x28_uk_lab), accuracy, "group_69_classes.txt")


choice = input("Enter 1 for SVM, 2 for CNN, 3 for both: ")

if choice == "1":
    svm(False)
elif choice == "2":
    cnn(False)
elif choice == "3":
    svm()
    cnn()
