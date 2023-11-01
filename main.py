from sklearn.model_selection import train_test_split

from SVM import SVMClassifier
from CNN import CNNClassifier
import numpy as np
import pandas as pd


def file_printer(prediction, estimated_accuracy, filename):
    result = np.append(estimated_accuracy, prediction)
    pd.DataFrame(result).to_csv(filename, index=False, header=False)


# Load data
mnist_28x28 = np.load('data/mnist_28x28_train.npy')
labels = np.load('data/mnist_labels.npy')

# Split data for CNN
mnist_28x28_train, mnist_28x28_test, labels_train, labels_test = train_test_split(mnist_28x28, labels,
                                                                                  test_size=0.2, random_state=42)

# SVM
svm = SVMClassifier(mnist_28x28, labels)
svc_param_grid = {
    'C': [2.745],
    'kernel': ['rbf', ],
    'degree': [0, 1, 2],
    'gamma': ['scale', 'auto'],
    'random_state': [42]
}
print("SVC OUTPUT:")
print("Best hyperparameters: " + svm.fine_tune_svm(param_grid=svc_param_grid).__str__())
print("Accuracy: " + svm.cross_val_accuracy().__str__())

# CNN
print("\n-------------------------------------------------------------")
print("CNN OUTPUT:")

cnn = CNNClassifier()
cnn.train(mnist_28x28_train, labels_train, epochs=9)
print("Accuracy: " + cnn.evaluate(mnist_28x28_test, labels_test).__str__())
