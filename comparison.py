from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import tree
import time
import ssl
from memory_profiler import memory_usage
import joblib
import os
import platform

# Supress Tensorflow logging messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Function to train the model
def train_model(model, x_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    return model.fit(x_train, y_train)


# Function that has the model make predictions
def model_predict(model, x_test: np.ndarray) -> np.ndarray:
    return model.predict(x_test)


# Function to check if the program has run before
def has_run_before():
    marker_file_path = os.path.expanduser("~/.marker_file")
    if os.path.exists(marker_file_path):
        return True
    else:
        with open(marker_file_path, "w") as f:
            f.write(
                "This is a marker file to indicate that the program has run before."
            )
        return False


# Function to prompt the user if they would like to attempt a quick execution of the program
def quick_execution_prompt() -> str:
    while True:
        choice = input(
            "Would you like to attempt a quick execution of the program? (y/n): "
        )
        if choice in ["y", "n"]:
            return choice
        else:
            print("Invalid input. Please try again.")


# Function to create a new directory to save the machine learning models
def create_new_directory():
    choice = "n"
    dir_end = input(
        "Please enter the folder name where you would like to save the machine learning models: "
    )
    save_dir = os.path.join(os.path.dirname(os.getcwd()), dir_end)
    os.makedirs(save_dir)
    return choice, save_dir


# Function to unpack and normalize the CIFAR-10 dataset
def unpack_and_normalize_data():
    print("Downloading the CIFAR-10 dataset...")

    # Unpacking the cifar10 dataset into two tuples (training images, training labels) and (testing images, testing labels)
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    print("Formatting the data...")

    # Ensure pixel values are floats to support decimal division
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")

    # Normalization
    x_train /= 255.0
    x_test /= 255.0

    # Reshaping the image arrays to 2d arrays since sklearn expects 2d arrays
    nsamples, nx, ny, nrgb = x_train.shape
    x_train2 = x_train.reshape((nsamples, nx * ny * nrgb))
    nsamples, nx, ny, nrgb = x_test.shape
    x_test2 = x_test.reshape((nsamples, nx * ny * nrgb))

    # Reshaping the labels to 1d arrays since sklearn expects 1d arrays
    y_train = y_train.ravel()
    y_test = y_test.ravel()

    return x_train2, y_train, x_test2, y_test


# Function to ask the user for the directory where the machine learning models are saved
def ask_directory():
    while True:
        directory = input(
            'Please enter the folder name where you saved the machine learning models: (if you don\'t know, type "idk"): '
        )
        full_directory = os.path.join(os.path.dirname(os.getcwd()), directory)
        if os.path.exists(full_directory):
            return full_directory
        elif directory == "idk":
            return directory
        else:
            print("Invalid directory. Please try again.")


# Function to find the appropriate directory to save the machine learning models
def find_directory(choice: str):
    print("Finding appropriate directory")

    # Paths to the machine learning models
    save_dirs = {
        "School Laptop": "/Users/MELLIPAR000/Downloads/machine_learning_models/",
        "PC": "C:/Users/parke/Downloads/machine-learning-models/",
        "Your Computer": "Your_Path_Here",
    }

    # Path to the machine learning models - replace with your own path
    if platform.system() == "Darwin" and os.path.exists(save_dirs["School Laptop"]):
        save_dir = save_dirs["School Laptop"]
    elif platform.system() == "Windows" and os.path.exists(save_dirs["PC"]):
        save_dir = save_dirs["PC"]
    elif os.path.exists(save_dirs["Your Computer"]):
        save_dir = save_dirs["Your Computer"]
    elif choice == "y":
        save_dir = ask_directory()
        if save_dir == "idk":
            choice, save_dir = create_new_directory()
    else:
        choice, save_dir = create_new_directory()
    return save_dir

# Function to create a bar graph
def create_special_bar_graph(ylabel, values, units, fontsize = 'large'):
    # Using the global variable num_plots to keep track of the number of subplots
    global num_plots

    # Creating a subplot for the bar graph
    plt.subplot(2, 3, num_plots)

    # Creating the bar graph
    plt.bar(0, values[0], 0.9)
    plt.bar(1, values[1], 0.9)
    plt.xlabel("Models")
    plt.ylabel(ylabel)
    plt.title(ylabel + " vs. Models")
    plt.xticks([0, 1], ["Random Forest", "Decision Tree"])

    # Annotating the bars with the values
    for index, value in enumerate(values):
        plt.text(index, value / 2, str(round(value, 2)) + ' ' + units, ha = 'center', va = 'bottom', fontsize = fontsize, fontweight = 'bold')
    
    # Incrementing the number of subplots
    num_plots += 1

def create_bar_graph(xlabel, ylabel, labels, values, units, fontsize = 'large'):
    # Creating the bar graph
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(ylabel + " vs. " + xlabel)
    plt.xticks(range(10), labels)

    # Annotating the bars with the values
    for index, value in enumerate(values):
        plt.bar(index, value, 0.9)
        plt.text(index, value / 2, str(round(value, 2)) + ' ' + units, ha = 'center', va = 'bottom', fontsize = fontsize, fontweight = 'bold')

# Disabling SSL certificate verification (only leave this uncommented when on a school-managed device)
ssl._create_default_https_context = ssl._create_unverified_context

# In place to prevent multiprocessing errors due to the memory profiler
if __name__ == "__main__":

    param_grid = {'max_depth': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]}

    # Asking the user if they would like to attempt a quick execution of the program if they have run the program before
    if has_run_before():
        choice = quick_execution_prompt()
        save_dir = find_directory(choice)
    else:
        choice, save_dir = create_new_directory()

    # Starting a program timer
    start_time = time.time()

    # Unpacking and normalizing the data
    x_train, y_train, x_test, y_test = unpack_and_normalize_data()

    # Path to the random forest model and training time
    rf_path = os.path.join(save_dir, "random_forest_model.joblib")
    rf_training_time_path = os.path.join(save_dir, "random_forest_training_time.txt")
    rf_training_memory_path = os.path.join(
        save_dir, "random_forest_training_memory.txt"
    )

    # Checking if the random forest model already exists and if the user wants to use it
    if (
        choice == "y"
        and os.path.exists(rf_path)
        and os.path.exists(rf_training_time_path)
        and os.path.exists(rf_training_memory_path)
    ):

        print("Loading the Random Forest model...")

        # Loading the random forest model
        rf = joblib.load(rf_path)

        # Loading the training time
        with open(rf_training_time_path, "r") as f:
            rf_train_time = float(f.read())

        # Loading the training memory
        with open(rf_training_memory_path, "r") as f:
            rf_training_memory = float(f.read())

    else:

        print("Training the Random Forest model...")

        # Creating an instance of the RandomForestClassifier
        rf = RandomForestClassifier() #n_jobs = -1
        grid_search_rf = GridSearchCV(rf, param_grid, cv = 5, n_jobs = -1)

        # Starting a timer for the RandomForestClassifier training
        rf_train_start_time = time.time()

        # Training the RandomForestClassifier
        train_model(grid_search_rf, x_train, y_train)

        # Ending the RandomForestClassifier training timer
        rf_train_time = time.time() - rf_train_start_time

        best_depth_rf = grid_search_rf.best_params_['max_depth']
        print(f"Optimal depth for Random Forest: {best_depth_rf}")

        print("Logging the Random Forest training memory usage...")

        # Measuring memory usage in MiB and converting to MB
        rf_training_memory = memory_usage((train_model, (rf, x_train, y_train)), max_usage=True) * 1.04858

        # Saving the RandomForestClassifier model
        joblib.dump(rf, rf_path)

        # Saving the training time
        with open(rf_training_time_path, "w") as f:
            f.write(str(rf_train_time))

        # Saving the training memory
        with open(rf_training_memory_path, "w") as f:
            f.write(str(rf_training_memory))

    # Getting the random forest model size
    rf_model_size = os.path.getsize(rf_path) / (1024 * 1024)

    print("Evaluating the Random Forest model...")

    # Starting a timer for the RandomForestClassifier prediction
    rf_prediction_start_time = time.time()

    # Predicting the test set
    y_pred_rf = model_predict(rf, x_test)

    # Ending the RandomForestClassifier prediction timer
    rf_prediction_end_time = time.time()

    # Measuring memory usage for the prediction in MiB and converting to MB
    rf_prediction_memory = memory_usage((model_predict, (rf, x_test)), max_usage=True) * 1.04858

    rf_confusion_matrix = confusion_matrix(y_test, y_pred_rf)

    # Printing the RandomForestClassifier performance metrics
    print("\nOverall Random Forest Accuracy:", round(accuracy_score(y_pred_rf, y_test) * 100, 2), "%\n",)
    print("Random Forest Classification Report:\n", classification_report(y_pred_rf, y_test),)
    print("Random Forest Confusion Matrix:\n", rf_confusion_matrix)
    print(f"\nRandom Forest Training Time: {round(rf_train_time, 2)} seconds")
    print(f"Random forest Prediction Time: {round((rf_prediction_end_time - rf_prediction_start_time) * 1000, 2)} milliseconds")
    print(f"Random Forest Training Memory: {round(rf_training_memory, 2)} Mb")
    print(f"Random Forest Prediction Memory: {round(rf_prediction_memory, 2)} Mb")
    print(f"Random Forest Model Size: {round(rf_model_size, 2)} Mb")

    # Path to the decision tree model
    dtc_path = os.path.join(save_dir, "decision_tree_model.joblib")
    dtc_training_time_path = os.path.join(save_dir, "decision_tree_training_time.txt")
    dtc_training_memory_path = os.path.join(save_dir, "decision_tree_training_memory.txt")

    # Checking if the decision tree model already exists and if the user wants to use it
    if (
        choice == "y"
        and os.path.exists(dtc_path)
        and os.path.exists(dtc_training_time_path)
        and os.path.exists(dtc_training_memory_path)
    ):

        print("Loading the Decision Tree model...")

        # Loading the decision tree model
        dtc = joblib.load(dtc_path)

        # Loading the training time
        with open(dtc_training_time_path, "r") as f:
            dtc_train_time = float(f.read())

        # Loading the training memory
        with open(dtc_training_memory_path, "r") as f:
            dtc_training_memory = float(f.read())

    else:

        print("Training the Decision Tree model...")

        # Creating an instance of the DecisionTreeClassifier
        dtc = DecisionTreeClassifier()
        grid_search_dtc = GridSearchCV(dtc, param_grid, cv = 5, n_jobs = -1)

        # Starting a timer for the DecisionTreeClassifier training
        dtc_train_start_time = time.time()

        train_model(grid_search_dtc, x_train, y_train)

        # Ending the DecisionTreeClassifier training timer
        dtc_train_time = time.time() - dtc_train_start_time

        best_depth_dtc = grid_search_dtc.best_params_['max_depth']
        print(f"Optimal depth for Decision Tree: {best_depth_dtc}")

        print("Logging the Decision Tree training memory usage...")

        # Measuring memory usage in MiB and converting to MB
        dtc_training_memory = memory_usage((train_model, (dtc, x_train, y_train)), max_usage = True) * 1.04858

        # Saving the DecisionTreeClassifier model
        joblib.dump(dtc, dtc_path)

        # Saving the training time
        with open(dtc_training_time_path, "w") as f:
            f.write(str(dtc_train_time))

        # Saving the training memory
        with open(dtc_training_memory_path, "w") as f:
            f.write(str(dtc_training_memory))

    # Getting the random forest model size
    dtc_model_size = os.path.getsize(dtc_path) / (1024 * 1024)

    print("Evaluating the Decision Tree model...")

    # Starting a timer for the DecisionTreeClassifier prediction
    dtc_prediction_start_time = time.time()

    # Predicting the test set
    y_pred_dtc = model_predict(dtc, x_test)

    # Ending the DecisionTreeClassifier prediction timer
    dtc_prediction_end_time = time.time()

    # Measuring memory usage for the prediction in MiB and converting to MB
    dtc_prediction_memory = memory_usage((model_predict, (dtc, x_test)), max_usage = True) * 1.04858

    dtc_confusion_matrix = confusion_matrix(y_test, y_pred_dtc)

    # Printing the DecisionTreeClassifier performance metrics
    print("\nOverall Decision Tree Accuracy:", round(accuracy_score(y_pred_dtc, y_test) * 100, 2), "%\n")
    print("Decision Tree Classification Report:\n", classification_report(y_pred_dtc, y_test))
    print("Decision Tree Confusion Matrix:\n", dtc_confusion_matrix)
    print(f"\nDecision Tree Training Time: {round(dtc_train_time, 2)} seconds")
    print(f"Decision Tree Prediction Time: {round((dtc_prediction_end_time - dtc_prediction_start_time) * 1000, 2)} milliseconds")
    print(f"Decision Tree Model Size: {round(dtc_model_size, 2)} Mb")
    print(f"Decision Tree Training Memory: {round(dtc_training_memory, 2)} Mb")
    print(f"Decision Tree Prediction Memory: {round(dtc_prediction_memory, 2)} Mb")

    print("Decision Tree Depth", dtc.get_depth())
    print("Random Forest Max Decision Tree Depth", max([rf.estimators_[i].get_depth() for i in range(100)]))
'''
    # Creating lists to store the model sizes, training times, prediction accuracies, predictions per second, training memories, and prediction memories
    model_sizes = [rf_model_size, dtc_model_size]
    training_times = [rf_train_time, dtc_train_time]
    prediction_accuracies = [accuracy_score(y_pred_rf, y_test) * 100, accuracy_score(y_pred_dtc, y_test) * 100]
    prediction_times = [rf_prediction_end_time - rf_prediction_start_time, dtc_prediction_end_time - dtc_prediction_start_time]
    predictions_per_sec = [10 / i for i in prediction_times]
    training_memories = [rf_training_memory, dtc_training_memory]
    prediction_memories = [rf_prediction_memory, dtc_prediction_memory]

    # Ending the program timer
    end_time = time.time()

    # Calculating and printing the execution time
    print(f"\nExecution Time: {end_time - start_time} seconds")

    # Creating a figure to display the bar graphs
    plt.figure()
    num_plots = 1

    # Creating bar graphs for the model sizes, training times, prediction accuracies, predictions per second, training memories, and prediction memories
    create_special_bar_graph("Model Size (MB)", model_sizes, 'MB')
    create_special_bar_graph("Training Time (seconds)", training_times, 'sec')
    create_special_bar_graph("Prediction Accuracy (%)", prediction_accuracies, '%')
    create_special_bar_graph("# Predictions Per Second (in thousands)", predictions_per_sec, '* 10^3 pps', 'medium')
    create_special_bar_graph("Training Memory (MB)", training_memories, 'MB')
    create_special_bar_graph("Prediction Memory (MB)", prediction_memories, 'MB')

    # Adjusting the spacing between the subplots
    plt.subplots_adjust(hspace = 0.3, wspace = 0.3)

    # Displaying the bar graphs
    plt.show()

    # Labels with index corresponding to the image class
    labels = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]

    # Creating lists to store the model accuracies for each image class
    rf_class_accuracies = rf_confusion_matrix.diagonal() / 10
    dtc_class_accuracies = dtc_confusion_matrix.diagonal() / 10

    # Creating a figure to display the bar graph
    plt.figure()

    # Creating a bar graph for the random forest classifier
    create_bar_graph("Image Class", "Accuracy", labels, rf_class_accuracies, '%')

    # Displaying the bar graph
    plt.show()

    # Creating a figure to display the bar graph
    plt.figure()

    # Creating a bar graph for the decision tree classifier
    create_bar_graph("Image Class", "Accuracy", labels, dtc_class_accuracies, '%')
    
    # Displaying the bar graph
    plt.show()

    # Annotating the bars with the values
    for index, value in enumerate(rf_class_accuracies):
        plt.text(index, value / 2, str(round(value, 2)) + '%', ha = 'center', va = 'bottom', fontsize = 'medium', fontweight = 'bold')

    # Displaying the decision tree
    tree.plot_tree(dtc, filled = True, fontsize = 6)
    plt.show()

    # Plot first decision tree in the random forest classifier
    tree.plot_tree(rf.estimators_[0], filled = True, fontsize = 6)
    plt.show()'''
    