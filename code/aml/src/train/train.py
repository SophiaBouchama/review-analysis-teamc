import argparse

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

import mlflow

parser = argparse.ArgumentParser()
parser.add_argument("--training_data", type=str, help="Path of prepped data")
parser.add_argument("--registered_model_name", type=str, help="model name")
args = parser.parse_args()

df = pd.read_csv(args.training_data, index_col="Id")

mlflow.start_run()
mlflow.sklearn.autolog()

print(df.head())

mlflow.log_metric("nb of features", df.shape[1])
mlflow.log_metric("nb of samples", df.shape[0])

X = df.apply(lambda x : x["Summary"] + " " + x["Text"], axis = 1) # nlp cleaning should be done beforehand
y = df.Score

# 70 / 20 / 10
X_train, X_test1, y_train, y_test1 = train_test_split(X, y, test_size=0.3, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test1, y_test1, test_size=0.33, random_state=42)

# Use count vectorizer
vect = CountVectorizer()

X_train_vect = vect.fit_transform(X_train)
X_test_vect = vect.transform(X_test)

# Multinomial NB
clf = MultinomialNB()
clf.fit(X_train_vect, y_train)

y_pred = clf.predict(X_test_vect)

print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

# REGISTER MODEL
mlflow.sklearn.log_model(
    sk_model=clf,
    registered_model_name=args.registered_model_name,
    artifact_path=args.registered_model_name
)

mlflow.end_run()