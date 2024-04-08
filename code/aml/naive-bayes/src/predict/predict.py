import argparse

import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import dill

import mlflow

parser = argparse.ArgumentParser()
parser.add_argument("--training_data_folder", type=str, help="Folder path of prepped data")
parser.add_argument("--training_data_name", type=str, help="Name of the prepped data")
parser.add_argument("--registered_model_name", type=str, help="model name")
parser.add_argument("--alpha", type=float, help="alpha")
parser.add_argument("--test_data", type=str, help="Path to test data")
parser.add_argument("--model_output", type=str, help="Path of output model")

args = parser.parse_args()

# load model and data from previous steps

mlflow.start_run()
mlflow.sklearn.autolog()

y_pred = clf.predict(X_val_vect)
y_pred_test = clf.predict(X_test_vect)

print(classification_report(y_val, y_pred))
mlflow.log_metric("val accuracy", accuracy_score(y_val, y_pred))
mlflow.log_metric("test accuracy", accuracy_score(y_test, y_pred_test))





