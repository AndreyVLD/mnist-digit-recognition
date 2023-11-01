import numpy as np
from MNISTClassifier import SVMClassifier

mnist_28x28_train = np.load("data/mnist_28x28_train.npy")
mnist_28x28_labels = np.load("data/mnist_labels.npy")
labels = mnist_28x28_labels

svm = SVMClassifier(mnist_28x28_train, labels)
svc_param_grid = {
    'C': [0.1, 1, 2.745, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'degree': [0, 1, 2, 3, 4, 5],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    'random_state': [42]
}
print(svm.fine_tune_svm(param_grid=svc_param_grid))
print(svm.cross_val_accuracy())
