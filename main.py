import numpy as np
from MNISTClassifier import SVMClassifier

mnist_28x28_train = np.load("data/mnist_28x28_train.npy")
mnist_28x28_labels = np.load("data/mnist_labels.npy")
labels = mnist_28x28_labels

svm = SVMClassifier(mnist_28x28_train, labels)
print(svm.fine_tune_svm())
print(svm.cross_val_accuracy())
svm.heatmap()
svm.plot_images()
