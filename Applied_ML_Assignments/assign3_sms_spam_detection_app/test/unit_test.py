import os, yaml, pickle, joblib
from src.score import score
import numpy as np


##########################################################################################################################


# extract the parameters in configs
config_file_path = os.path.join('config', 'test_config', 'unit.yaml')

with open(config_file_path) as fh:
    config = yaml.load(fh, Loader=yaml.FullLoader)

# extract the parameters

# extract the input texts
spam_text = config['input_text']['spam']
ham_text = config['input_text']['ham']

# extract the vectorizer
tfidf = pickle.load(open(config['tfidf_save_path'], 'rb'))
# extract the classifier
best_clf = joblib.load(config['best_clf_save_path'])
# extract the threshold
default_threshold = config['threshold']

# extract the output of score function
is_spam, proba = score(
    input_text=spam_text,
    vectorizer=tfidf,
    classifier=best_clf,
    threshold=default_threshold
)


class TestClass:

    # check if score function returns values properly
    def test_smoke(self):
        assert (is_spam != None)
        assert (proba != None)
    
    # check if input/output types are valid
    def test_format(self):
        assert type(spam_text) == str
        assert type(default_threshold) == float
        assert type(is_spam) == bool
        assert type(proba) == np.float64

    # check if output label = 0/1
    def test_pred_value(self):
        assert (is_spam == True or is_spam == False)
    
    # check if propensity lies in [0,1]
    def test_prop_value(self):
        assert ((proba >= 0) and (proba <= 1))

    # sanity check: if threshold = 0, prediction = `True`
    def test_pred_th0(self):
        label, _ = score(
            input_text=spam_text,
            vectorizer=tfidf,
            classifier=best_clf,
            threshold=0
        )
        assert (label == True)

    # sanity check: if threshold = 1, prediciton = `False`
    def test_pred_th1(self):
        label, _ = score(
            input_text=spam_text,
            vectorizer=tfidf,
            classifier=best_clf,
            threshold=1
        )
        assert (label == False)
    
    # testing the default spam message
    def test_spam(self):
        label, _ = score(
            input_text=spam_text,
            vectorizer=tfidf,
            classifier=best_clf,
            threshold=default_threshold
        )
        assert (label == True)

    # testing the default non-spam (ham) message
    def test_ham(self):
        label, _ = score(
            input_text=ham_text,
            vectorizer=tfidf,
            classifier=best_clf,
            threshold=default_threshold
        )
        assert (label == False)


##########################################################################################################################


# to produce coverage report run the following command in terminal:
# coverage run -m pytest test/unit_test.py && coverage report -m > coverage_reports/unit_test_coverage.txt