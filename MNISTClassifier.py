import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt


# Reshape the images to 1D array and scale the values to [0, 1]
def reshape_and_scale(data):
    reshaped_images_28x28 = [image.reshape(-1) for image in data]
    scaler = MinMaxScaler()
    return scaler.fit_transform(reshaped_images_28x28)


class SVMClassifier:
    def __init__(self, x, y):
        self.total_observations = reshape_and_scale(x)
        self.labels = y

        x_train, x_test, y_train, y_test = train_test_split(self.total_observations, self.labels, test_size=0.2,
                                                            random_state=42)

        self.X_train = x_train
        self.y_train = y_train
        self.X_test = x_test
        self.y_test = y_test
        self.class_labels = np.unique(y_train)
        self.svc = None

    def fine_tune_svm(self, param_grid=None):
        print("Fine-tuning the SVC with GridSearchCV...")

        svc_param_grid = {
            'C': [2.74, 2.75, 2.76],
            'kernel': ['linear', 'rbf'],
            'degree': [0, 1, 2, 3, 4],
            'gamma': ['scale', 'auto'],
            'random_state': [42]
        }
        if param_grid is not None:
            svc_param_grid = param_grid

        svc = SVC()
        svc_grid_search = GridSearchCV(estimator=svc, param_grid=svc_param_grid, scoring='accuracy', cv=5, n_jobs=-1)
        svc_grid_search.fit(self.X_train, self.y_train)

        self.svc = SVC(C=svc_grid_search.best_params_['C'], degree=svc_grid_search.best_params_['degree'],
                       kernel=svc_grid_search.best_params_['kernel'], gamma=svc_grid_search.best_params_['gamma'],
                       random_state=svc_grid_search.best_params_['random_state'])

        print("Fitting the SVC with the best parameters to the data...")
        self.svc.fit(self.X_train, self.y_train)
        return svc_grid_search.best_params_

    def cross_val_accuracy(self, x=None, y=None, num_folds=5):
        print("Cross validation to get the expected accuracy...")
        if self.svc is not None:
            if x is None or y is None:
                x = self.total_observations
                y = self.labels

            cross_val_scores = cross_val_score(self.svc, x, y, cv=num_folds, scoring='accuracy')
            return cross_val_scores.mean()
        else:
            raise ValueError("SVM model not trained. Run fine_tune_svm() first.")

    def confusion_matrix(self, x_test, y_test):
        if self.svc is not None:
            y_pred = self.svc.predict(x_test)
            return confusion_matrix(y_test, y_pred)
        else:
            raise ValueError("SVM model not trained. Run fine_tune_svm() first.")

    def heatmap(self, x_test=None, y_test=None):
        if self.svc is not None:
            if x_test is None or y_test is None:
                x_test = self.total_observations
                y_test = self.labels

            y_pred = self.svc.predict(x_test)
            conf_matrix = confusion_matrix(y_test, y_pred)
            plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Confusion Matrix')
            plt.colorbar()
            tick_marks = np.arange(len(self.class_labels))

            plt.xticks(tick_marks, self.class_labels, rotation=45)
            plt.yticks(tick_marks, self.class_labels)

            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')

            for i in range(len(self.class_labels)):
                for j in range(len(self.class_labels)):
                    plt.text(j, i, str(conf_matrix[i, j]), ha='center', va='center', color='red')

            plt.show()
        else:
            raise ValueError("SVM model not trained. Run fine_tune_svm() first.")

    def overall_accuracy(self, x_test=None, y_test=None):
        if self.svc is not None:
            if x_test is None or y_test is None:
                x_test = self.X_test
                y_test = self.y_test

            y_pred = self.svc.predict(x_test)
            return accuracy_score(y_test, y_pred)
        else:
            raise ValueError("SVM model not trained. Run fine_tune_svm() first.")

    def predict(self, x_test):
        if self.svc is not None:
            return self.svc.predict(x_test)
        else:
            raise ValueError("SVM model not trained. Run fine_tune_svm() first.")

    def plot_images(self, images=None, labels=None, num_images=5):
        if num_images > 0:
            if images is None or labels is None:
                images = self.total_observations
                labels = self.labels
            for i in range(num_images):
                plt.figure(figsize=(2, 2))
                plt.imshow(images[i].reshape(28, 28), cmap="gray")
                plt.title(f"Actual: {labels[i]}")
                plt.show()
        else:
            print("Specify a positive number of images to plot.")
