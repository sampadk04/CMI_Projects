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

# for saving and loading models
import pickle

# suppress warnings
import warnings
warnings.filterwarnings("ignore")

##########################################################################################################################


# extract the parameters in configs
config_file_path = os.path.join('config', 'score.yaml')

with open(config_file_path) as fh:
    config = yaml.load(fh, Loader=yaml.FullLoader)


# predict if a given text is spam/ham and it's corresponding probability
def score(input_text:str, vectorizer, classifier, threshold:float=0.5):
    # put the text into an array
    text_arr = np.array([input_text])
    
    # vectorize the input text
    text_vectorized = vectorizer.transform(text_arr)
    # predict the probability score
    proba = classifier.predict_proba(text_vectorized)[:, 1][0]

    is_spam = False
    if proba >= threshold:
        is_spam = True

    return is_spam, proba


##########################################################################################################################


if __name__=='__main__':
    # extract the input text
    input_text = config['input_text']

    print("Input Text:")
    print(input_text)

    # extract the vectorizer
    tfidf = pickle.load(open(config['tfidf_save_path'], 'rb'))

    # extract the classifier
    best_clf = joblib.load(config['best_clf_save_path'])

    # extract the threshold
    threshold = config['threshold']

    is_spam, proba = score(
        input_text=input_text,
        vectorizer=tfidf,
        classifier=best_clf,
        threshold=threshold
    )

    print("Spam Status:", is_spam)
    print("Propensity:", proba.round(3))