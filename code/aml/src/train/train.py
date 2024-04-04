import argparse

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from utils import doc_vectorizer

import xgboost
from sklearn.ensemble import GradientBoostingClassifier

import mlflow

parser = argparse.ArgumentParser()
parser.add_argument("--training_data", type=str, help="Path of prepped data")
parser.add_argument("--registered_model_name", type=str, help="model name")
args = parser.parse_args()

print(args.training_data)
print(args.registered_model_name)

df = pd.read_csv(args.training_data)

mlflow.start_run()
mlflow.sklearn.autolog()

print(df.head())
print(df.shape)

mlflow.log_metric("nb of features", df.shape[1])
mlflow.log_metric("nb of samples", df.shape[0])

df.dropna(inplace=True)

X = df[["CleanedText", "Score"]]
y = df.Score

data_train, data_val_test, Y_train, Y_val_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=X['Score']
)

data_val, data_test, Y_val, Y_test = train_test_split(
    data_val_test, Y_val_test, test_size=0.333, random_state=42, stratify=data_val_test['Score']
)

data_train = data_train['CleanedText']

X_train, X_val, X_test = doc_vectorizer(data_train, data_val, data_test, "doc2vec", {'vector_size':200, 'window':5, 'min_count':1, 'workers':4, 'epochs':20})

X_train_vect = vect.fit_transform(X_train)
X_val_vect = vect.transform(X_val)

model.fit(X_train, Y_train)

y_pred = clf.predict(X_val_vect)

print(classification_report(y_val, y_pred))
print(accuracy_score(y_val, y_pred))

# REGISTER MODEL
mlflow.sklearn.log_model(
    sk_model=model,
    registered_model_name=args.registered_model_name,
    artifact_path=args.registered_model_name
)

mlflow.end_run()