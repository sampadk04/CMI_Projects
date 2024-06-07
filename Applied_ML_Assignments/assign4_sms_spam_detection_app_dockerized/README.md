# SMS Spam Detection App

This is a simple Flask web application for detecting whether an SMS is spam or not.

The SMS spam detection model was trained using the `SMS Spam Collection Dataset`. The model was built using scikit-learn and achieved an accuracy of about 98% on the test dataset.

The pre-trained model is included in the repository as a `joblib` file (`models/best_clf.joblib`). 

When the Flask app is started, the model is loaded into memory and used for making predictions.