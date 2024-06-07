import os

# for data handling
import pandas as pd
import numpy as np

# for plotting
import matplotlib.pyplot as plt

# for evaluation purposes
from sklearn.metrics import classification_report

# for saving the models
import joblib

# for configs
import yaml

# suppress warnings
import warnings
warnings.filterwarnings("ignore")

# loading the train, val, test data from the split
from train import load_train_val_test

##########################################################################################################################


# extract the parameters in configs
config_file_path = os.path.join('config', 'evaluate.yaml')

with open(config_file_path) as fh:
    config = yaml.load(fh, Loader=yaml.FullLoader)


# this prints classification report of the classifier on the 'Test' data
def print_classification_report(X_test, y_test, classifier):
    # prediction on X_test
    y_test_hat = classifier.predict(X_test)

    # print the classification report
    print("Classification Report for", classifier.__class__.__name__, ": ")
    print(classification_report(y_test, y_test_hat))


##########################################################################################################################


if __name__=='__main__':
    # use the loader function
    _, _, X_test, _, _, y_test = load_train_val_test(config['train_val_test_save_paths'])

    # print data split info
    print("Test Data Shape:", X_test.shape)

    # loading the models and printing their classification reports on the test data

    # Logistic Regression
    best_Logit = joblib.load(config['model_save_paths']['Logit_save_path'])
    print_classification_report(X_test, y_test, best_Logit)

    # Random Forest Classifier
    best_RFC = joblib.load(config['model_save_paths']['RFC_save_path'])
    print_classification_report(X_test, y_test, best_RFC)

    # Gradient Boosting Classifier
    best_GBC = joblib.load(config['model_save_paths']['GBC_save_path'])
    print_classification_report(X_test, y_test, best_GBC)

    # AdaBoost Classifier
    best_ABC = joblib.load(config['model_save_paths']['ABC_save_path'])
    print_classification_report(X_test, y_test, best_ABC)

    # since, Logistic Regression model seems to perform the best overall (good accuracy and precision score)
    print("Saving Best Classifier...")
    best_clf = best_Logit
    joblib.dump(best_clf, config['best_clf_save_path'])
    print("Saved!")