import os

# for data handling
import pandas as pd
import numpy as np

# for plotting
import matplotlib.pyplot as plt

# for data splitting
from sklearn.model_selection import train_test_split

# for data pre-processing
from sklearn.feature_extraction.text import TfidfVectorizer

# for configs
import yaml

# for saving models
import pickle

# suppress warnings
import warnings
warnings.filterwarnings("ignore")

##########################################################################################################################


# extract the parameters in configs
config_file_path = os.path.join('config', 'prepare.yaml')

with open(config_file_path) as fh:
    config = yaml.load(fh, Loader=yaml.FullLoader)


# loading the data
def load_data(file_path):
    df = pd.read_csv(file_path, delimiter='\t', header=None, names=['label', 'text'])
    # convert labels to binary int 0/1
    df['label'] = df['label'].map({'ham':0, 'spam':1})
    return df


# preprocess data
def preprocess_data(df, vectorizer_save_path):
    # convert a collection of raw documents to a matrix of TF-IDF features
    
    # extract features and labels
    features = df['text'].copy()
    labels = df['label'].copy()

    # initialize the vectorizer
    TfVectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)

    # fit the Vectorizer on the data
    tfidf = TfVectorizer.fit(features)

    # transform the features
    features = TfVectorizer.transform(features)

    # save the vectorizer for later use
    pickle.dump(tfidf, open(vectorizer_save_path, 'wb'))

    # to load and use this model into the variable 'vectorizer' follow:
    '''
    vectorizer = pickle.load(open(vectorizer_save_path, 'rb'))
    vectorizer.transform(features)
    '''
    
    # convert from scipy sparse matrix to pandas dataframe
    features = pd.DataFrame.sparse.from_spmatrix(features)

    return features, labels


# splitting the data
def train_val_test_split(features, labels, random_state=None):
    # splitting into train, val, test
    
    # split into test and non-test
    X_non_test, X_test, y_non_test, y_test = train_test_split(features, labels, test_size=0.15, random_state=random_state)

    # split into train and val
    X_train, X_val, y_train, y_val = train_test_split(X_non_test, y_non_test, test_size=0.2, random_state=random_state)

    return X_train, X_val, X_test, y_train, y_val, y_test


# saving the data
def save_train_val_test_data(features, labels, train_val_test_save_paths, random_state):
    # extract train, test, val
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(features, labels, random_state=random_state)
    
    # save train, val, test data as .csv files
    train_data = pd.concat([X_train, y_train], axis=1)
    val_data = pd.concat([X_val, y_val], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    # save as .csv files to the savepaths
    train_data.to_csv(train_val_test_save_paths[0], index=False)
    val_data.to_csv(train_val_test_save_paths[1], index=False)
    test_data.to_csv(train_val_test_save_paths[2], index=False)

    print("Train, Val, Test data saved to:\n", train_val_test_save_paths)
    
    return None


##########################################################################################################################


if __name__=='__main__':

    # load data from raw file
    df = load_data(config['raw_data_path'])

    # preprocess this data
    features, labels = preprocess_data(
        df=df,
        vectorizer_save_path=config['tfidf_save_path']
    )

    # spltitting data into train, val, test
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(features, labels, random_state=config['random_state'])

    # print data split info
    print("Training Data Shape:", X_train.shape)
    print("Validation Data Shape:", X_val.shape)
    print("Test Data Shape:", X_test.shape)

    # saving the split data
    save_train_val_test_data(features, labels, train_val_test_save_paths=config['train_val_test_save_paths'], random_state=config['random_state'])