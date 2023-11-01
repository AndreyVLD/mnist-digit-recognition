import numpy as np
from keras.src.optimizers import Adam
from keras.src.preprocessing.image import ImageDataGenerator
from keras.src.utils import to_categorical
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.callbacks import EarlyStopping, LearningRateScheduler


def lr_schedule(x):
    return 0.001 * 0.9 ** x


class CNNClassifier:
    def __init__(self, lr=0.001, batch_size=32):
        self.batch_size = batch_size
        self.lr = lr
        self.model = self.build_model()
        self.history = None

    def build_model(self):
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer=Adam(learning_rate=self.lr),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def train(self, X_train, y_train, X_test, y_test, epochs=5):
        X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
        X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
        y_train = to_categorical(y_train, num_classes=10)
        y_test = to_categorical(y_test, num_classes=10)

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1,
                                     zoom_range=0.1)
        self.history = self.model.fit(datagen.flow(X_train, y_train, batch_size=self.batch_size),
                                      steps_per_epoch=len(X_train) / self.batch_size, epochs=epochs,
                                      validation_data=(X_test, y_test),
                                      callbacks=[LearningRateScheduler(lr_schedule), early_stopping])

    def evaluate(self, X_test, y_test):
        X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
        y_test = to_categorical(y_test, num_classes=10)
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
