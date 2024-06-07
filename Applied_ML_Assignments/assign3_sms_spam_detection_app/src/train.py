import os

# for data handling
import pandas as pd
import numpy as np

# for plotting
import matplotlib.pyplot as plt

# for classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# for evaluation purposes
from sklearn.metrics import confusion_matrix, precision_recall_curve, precision_score, recall_score, classification_report, ConfusionMatrixDisplay, make_scorer

# for custom gridsearch
from itertools import product

# for oversampling
from imblearn.over_sampling import SMOTE, RandomOverSampler

# for keeping track of loops
from tqdm import tqdm

# for saving the models
import joblib

# for configs
import yaml

# suppress warnings
import warnings
warnings.filterwarnings("ignore")


##########################################################################################################################


# extract the parameters in configs
config_file_path = os.path.join('config', 'train.yaml')

with open(config_file_path) as fh:
    config = yaml.load(fh, Loader=yaml.FullLoader)


# loading the train, val, test data from the split
def load_train_val_test(train_val_test_save_paths, oversampler=None):
    # extract the dataframes
    train_data = pd.read_csv(train_val_test_save_paths[0])
    val_data = pd.read_csv(train_val_test_save_paths[1])
    test_data = pd.read_csv(train_val_test_save_paths[2])
    
    # split the data into features, labels
    y_train = train_data['label']
    X_train = train_data.drop('label', axis=1)

    # oversample the training data
    if oversampler:
        X_train, y_train = oversampler.fit_resample(X_train, y_train)


    y_val = val_data['label']
    X_val = val_data.drop('label', axis=1)

    y_test = test_data['label']
    X_test = test_data.drop('label', axis=1)

    return X_train, X_val, X_test, y_train, y_val ,y_test


# train model by grid-searching over the param_grid
def train_model(X_train, X_val, y_train, y_val, classifier, param_grid):
    
    # init best model
    best_model = classifier
    
    # best precision
    best_precision = 0.0

    # make param_list by considering set products of params
    param_list = list(product(*param_grid.values()))

    for i in tqdm(range(len(param_list))):
        param = param_list[i]
        param_dict = dict(zip(param_grid.keys(), param))
        
        # init model with these params
        model = classifier.set_params(**param_dict)
        # fit the model on train data
        model.fit(X_train, y_train)

        # evaluate the model on val data
        y_val_hat = model.predict(X_val)
        # calculate precision
        current_precision = precision_score(y_val, y_val_hat, average='micro')

        # update model, score based on val precision
        if current_precision > best_precision:
            best_precision = current_precision
            best_model = model
            
            print("Current Best Precision on Val: %.3f" % best_precision)
    
    # print the best classifier
    print("Overall Best Model:", best_model)
    print("Overall Best Precision on Val: %.3f" % best_precision)

    return best_model


##########################################################################################################################


if __name__=='__main__':

    # loading the train, val, test files
    
    # define an oversampler for class imbalance
    smote = SMOTE(random_state=config['smote_random_state'])

    # use the loader function
    X_train, X_val, X_test, y_train, y_val ,y_test = load_train_val_test(config['train_val_test_save_paths'], oversampler=smote)

    # print data split info
    print("Training Data Shape:", X_train.shape)
    print("Validation Data Shape:", X_val.shape)
    print("Test Data Shape:", X_test.shape)

    # training classifiers in a grid_search manner to maximize `precision`

    print("Beginning Training:")

    # Logistic Regression
    classifier = LogisticRegression()
    best_Logit = train_model(
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        classifier=classifier,
        param_grid=config['model_param_grids']['Logit_param_grid']
    )
    # saving the model
    joblib.dump(best_Logit, config['model_save_paths']['Logit_save_path'])

    # Random Forest Classifier
    classifier = RandomForestClassifier()
    best_RFC = train_model(
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        classifier=classifier,
        param_grid=config['model_param_grids']['RFC_param_grid']
    )
    # saving the model
    joblib.dump(best_RFC, config['model_save_paths']['RFC_save_path'])

    # Gradient Boosting Classifier
    classifier = GradientBoostingClassifier()
    best_GBC = train_model(
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        classifier=classifier,
        param_grid=config['model_param_grids']['GBC_param_grid']
    )
    # saving the model
    joblib.dump(best_GBC, config['model_save_paths']['GBC_save_path'])

    # AdaBoost Classifier
    classifier = AdaBoostClassifier()
    best_ABC = train_model(
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        classifier=classifier,
        param_grid=config['model_param_grids']['ABC_param_grid']
    )
    # saving the model
    joblib.dump(best_ABC, config['model_save_paths']['ABC_save_path'])