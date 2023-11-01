import numpy as np
import keras

from keras.models import Model, Sequential
from keras.layers import Dense
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten
from keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt
from keras.utils import to_categorical


def build_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=5, activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPool2D())
    model.add(Conv2D(64, kernel_size=5, activation='relu'))
    model.add(MaxPool2D())
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


class CNNClassifier:
    def __init__(self, batch_size=32):
        self.batch_size = batch_size
        self.model = build_model()
        self.history = None

    def train(self, X_train, y_train, X_test, y_test, epochs=5):
        X_train = X_train / 255.0
        X_test = X_test / 255.0
        X_train = X_train.reshape(-1, 28, 28, 1)
        X_test = X_test.reshape(-1, 28, 28, 1)
        Y_train = to_categorical(y_train, num_classes=10)
        Y_test = to_categorical(y_test, num_classes=10)

        annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** (x + epochs))
        self.history = self.model.fit(X_train, Y_train, batch_size=self.batch_size, epochs=epochs, verbose=1,
                                      validation_data=(X_test, Y_test), callbacks=[annealer])

    def evaluate(self, X_test, y_test):
        X_test = X_test.reshape(-1, 28, 28, 1)
        y_test = keras.utils.to_categorical(y_test, num_classes=10)
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test)
        return test_accuracy

    def predict(self, new_data):
        predictions = self.model.predict(new_data)
        predicted_classes = np.argmax(predictions, axis=1)
        return predicted_classes

    def plot_history(self):
        plt.plot(self.history.history['accuracy'], label='training_accuracy')
        plt.plot(self.history.history['val_accuracy'], label='validation_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0, 1])
        plt.legend(loc='lower right')
        plt.show()
