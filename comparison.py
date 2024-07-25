from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import time
import ssl

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
model = RandomForestClassifier()

# Training the RandomForestClassifier
model.fit(x_train2, y_train)

# Predicting the test set
y_pred_rf = model.predict(x_test2)

# Printing the RandomForestClassifier performance metrics
print('\nOverall Random Forest Accuracy:', accuracy_score(y_pred_rf, y_test) * 100, '%\n')
print('Random Forest Classification Report:\n', classification_report(y_pred_rf, y_test))
print('Random Forest Confusion Matrix:\n', confusion_matrix(y_pred_rf, y_test))

# Creating an instance of the DecisionTreeClassifier
dtc = DecisionTreeClassifier()

# Training the DecisionTreeClassifier
dtc.fit(x_train2, y_train)

# Predicting the test set
y_pred_dtc = dtc.predict(x_test2)

# Printing the DecisionTreeClassifier performance metrics
print('\nOverall Decision Tree Accuracy:', accuracy_score(y_pred_dtc, y_test) * 100, '%\n')
print('Decision Tree Classification Report:\n', classification_report(y_pred_dtc, y_test))
print('Decision Tree Confusion Matrix:\n', confusion_matrix(y_pred_dtc, y_test))

# Ending the program timer
end_time = time.time()

# Calculating and printing the execution time
print('\nExecution Time:', end_time - start_time, 'seconds')