from sklearn.model_selection import train_test_split

from mnist.SVM import SVMClassifier as SVM
from mnist.CNN import CNNClassifier as CNN
import numpy as np
import pandas as pd


def file_printer(prediction, estimated_accuracy, filename):
    result = np.append(estimated_accuracy, prediction)
    pd.DataFrame(result).to_csv(filename, index=False, header=False)


# Load data
mnist_28x28 = np.load('data/mnist_28x28_train.npy')
labels = np.load('data/mnist_labels.npy')
mnist_28x28_uk = np.load('data/mnist_28x28_unknown.npy')


# SVM
def svm():
    svm_classifier = SVM(mnist_28x28, labels)
    svc_param_grid = {
        'C': [2.745],
        'kernel': ['rbf', ],
        'degree': [0, 1, 2],
        'gamma': ['scale', 'auto'],
        'random_state': [42]
    }
    print("\n-------------------------------------------------------------")
    print("SVC OUTPUT:")
    print("Best hyperparameters: " + svm_classifier.fine_tune_svm(param_grid=svc_param_grid).__str__())
    print("Accuracy: " + svm_classifier.cross_val_accuracy().__str__())


# CNN
def cnn():
    # Split data for CNN
    mnist_28x28_train, mnist_28x28_test, labels_train, labels_test = train_test_split(mnist_28x28, labels,
                                                                                      test_size=0.2, random_state=42)
    print("\n-------------------------------------------------------------")
    print("CNN OUTPUT:")

    cnn_classifier = CNN(lr=0.01, batch_size=8)
    cnn_classifier.train(mnist_28x28_train, labels_train, mnist_28x28_test, labels_test, epochs=30)
    accuracy = cnn_classifier.evaluate(mnist_28x28_test, labels_test)
    cnn_classifier.plot_history()

    print("Accuracy: " + accuracy.__str__())
    file_printer(cnn_classifier.predict(mnist_28x28_uk), accuracy, "GUD97.txt")


choice = input("Enter 1 for SVM, 2 for CNN, 3 for both: ")

if choice == "1":
    svm()
elif choice == "2":
    cnn()
elif choice == "3":
    svm()
    cnn()
