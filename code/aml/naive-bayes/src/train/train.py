import argparse

import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

import mlflow

parser = argparse.ArgumentParser()
parser.add_argument("--training_data", type=str, help="Path of prepped data")
parser.add_argument("--registered_model_name", type=str, help="model name")
parser.add_argument("--alpha", type=float, help="alpha")
parser.add_argument("--test_data", type=str, help="Path to test data")
parser.add_argument("--model_output", type=str, help="Path of output model")

args = parser.parse_args()

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

# 70 / 20 / 10
X_train, X_test1, y_train, y_test1 = train_test_split(X, y, test_size=0.3, random_state=42, stratify=X['Score'])
X_test, X_val, y_test, y_val = train_test_split(X_test1, y_test1, test_size=0.33, random_state=42, stratify=X_test1['Score'])

# Remove Score column from features
X_train = X_train['CleanedText']
X_test = X_test['CleanedText']
X_val = X_val['CleanedText']

# Use count vectorizer
vect = CountVectorizer()

X_train_vect = vect.fit_transform(X_train)
X_val_vect = vect.transform(X_val)
X_test_vect = vect.transform(X_test)

# Multinomial NB
clf = MultinomialNB(alpha=args.alpha)
clf.fit(X_train_vect, y_train)

y_pred = clf.predict(X_val_vect)
y_pred_test = clf.predict(X_test_vect)

print(classification_report(y_val, y_pred))
mlflow.log_metric("val accuracy", accuracy_score(y_val, y_pred))
mlflow.log_metric("test accuracy", accuracy_score(y_test, y_pred_test))

# REGISTER MODEL
mlflow.sklearn.log_model(
    sk_model=clf,
    registered_model_name=args.registered_model_name,
    artifact_path=args.registered_model_name
)

mlflow.sklearn.save_model(clf, args.model_output)