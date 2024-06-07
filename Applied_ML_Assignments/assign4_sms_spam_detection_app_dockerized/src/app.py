import os, yaml, pickle, joblib
from score import score
#from src.score import score
from flask import Flask, jsonify, request


##########################################################################################################################


# extract the parameters in configs
config_file_path = os.path.join('config', 'app.yaml')

with open(config_file_path) as fh:
    config = yaml.load(fh, Loader=yaml.FullLoader)

app = Flask(__name__)

@app.route('/score', methods=["POST"])
def sms_spam_detector():
    if request.method == "POST":
        # extract the input text
        sms_text = request.form["sms_text"]

        print("Input Text:")
        print(sms_text)

        # extract the vectorizer
        tfidf = pickle.load(open(config['tfidf_save_path'], 'rb'))

        # extract the classifier
        best_clf = joblib.load(config['best_clf_save_path'])

        # extract the threshold
        threshold = config['threshold']

        # use the classifier to predict wheher the text is spam or not
        is_spam, proba = score(
            input_text=sms_text,
            vectorizer=tfidf,
            classifier=best_clf,
            threshold=threshold
        )

        print("Spam Status:", is_spam)
        print("Propensity", proba.round(3))

        output_dict = {
            "prediction":is_spam,
            "propensity":proba.round(3)
        }
    
    return jsonify(output_dict)


##########################################################################################################################


if __name__ == '__main__':
    app.run(debug=True)