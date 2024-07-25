from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import time
import ssl
from memory_profiler import memory_usage

def train_model(model, x_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    return model.fit(x_train, y_train)

def model_predict(model, x_test: np.ndarray) -> np.ndarray:
    return model.predict(x_test)

# Disabling SSL certificate verification (only leave this uncommented when on a school-managed device)
ssl._create_default_https_context = ssl._create_unverified_context

# Starting a program timer
start_time = time.time()

# Unpacking the cifar10 dataset into two tuples (training images, training labels) and (testing images, testing labels)
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Ensure pixel values are floats to support decimal division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalization
x_train /= 255.0
x_test /= 255.0

# Reshaping the image arrays to 2d arrays since sklearn expects 2d arrays
nsamples, nx, ny, nrgb = x_train.shape
x_train2 = x_train.reshape((nsamples, nx*ny*nrgb))
nsamples, nx, ny, nrgb = x_test.shape
x_test2 = x_test.reshape((nsamples, nx*ny*nrgb))

# Reshaping the labels to 1d arrays since sklearn expects 1d arrays
y_train = y_train.ravel()
y_test = y_test.ravel()

# Creating an instance of the RandomForestClassifier
rf = RandomForestClassifier()

# Starting a timer for the RandomForestClassifier training
rf_train_start_time = time.time()

# Training the RandomForestClassifier and measuring memory usage
rf_training_memory = memory_usage(train_model, (rf, x_train2, y_train))

# Ending the RandomForestClassifier training timer
rf_train_end_time = time.time()

# Starting a timer for the RandomForestClassifier prediction
rf_prediction_start_time = time.time()

# Predicting the test set
y_pred_rf = model_predict(rf, x_test2)

# Ending the RandomForestClassifier prediction timer
rf_prediction_end_time = time.time()

# Measuring memory usage for the prediction
rf_prediction_memory = memory_usage(model_predict, (rf, x_test2))

# Printing the RandomForestClassifier performance metrics
print('\nOverall Random Forest Accuracy:', accuracy_score(y_pred_rf, y_test) * 100, '%\n')
print('Random Forest Classification Report:\n', classification_report(y_pred_rf, y_test))
print('Random Forest Confusion Matrix:\n', confusion_matrix(y_pred_rf, y_test))
print(f'\nRandom Forest Training Time: {rf_train_end_time - rf_train_start_time} seconds')
print(f'Random forest Prediction Time: {rf_prediction_end_time - rf_prediction_start_time} seconds')
print(f'Random Forest Training Memory: {rf_training_memory} MB')
print(f'Random Forest Prediction Memory: {rf_prediction_memory} MB')

# Creating an instance of the DecisionTreeClassifier
dtc = DecisionTreeClassifier()

# Starting a timer for the DecisionTreeClassifier training
dtc_train_start_time = time.time()

# Training the DecisionTreeClassifier and measuring memory usage
dtc_training_memory = memory_usage(train_model, (dtc, x_train2, y_train))

# Ending the DecisionTreeClassifier training timer
dtc_train_end_time = time.time()

# Starting a timer for the DecisionTreeClassifier prediction
dtc_prediction_start_time = time.time()

# Predicting the test set
y_pred_dtc = model_predict(dtc, x_test2)

# Ending the DecisionTreeClassifier prediction timer
dtc_prediction_end_time = time.time()

# Measuring memory usage for the prediction
dtc_prediction_memory = memory_usage(model_predict, (dtc, x_test2))

# Printing the DecisionTreeClassifier performance metrics
print('\nOverall Decision Tree Accuracy:', accuracy_score(y_pred_dtc, y_test) * 100, '%\n')
print('Decision Tree Classification Report:\n', classification_report(y_pred_dtc, y_test))
print('Decision Tree Confusion Matrix:\n', confusion_matrix(y_pred_dtc, y_test))
print(f'\nDecision Tree Training Time: {dtc_train_end_time - dtc_train_start_time} seconds')
print(f'Decision Tree Prediction Time: {dtc_prediction_end_time - dtc_prediction_start_time} seconds')
print(f'Decision Tree Training Memory: {dtc_training_memory} MB')
print(f'Decision Tree Prediction Memory: {dtc_prediction_memory} MB')

# Ending the program timer
end_time = time.time()

# Calculating and printing the execution time
print(f'\nExecution Time: {end_time - start_time} seconds')