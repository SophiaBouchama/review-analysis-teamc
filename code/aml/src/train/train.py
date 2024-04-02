import argparse

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


parser = argparse.ArgumentParser()
parser.add_argument("--prep_data", type=str, help="Path of prepped data")
parser.add_argument("--registered_model_name", type=str, help="model name")

df = pd.read_csv(args.prep_data, index_col="Id")

mlflow.start_run()
mlflow.sklearn.autolog()

print(df.head())

mlflow.log_metric("nb of features", df.shape[1])
mlflow.log_metric("nb of samples", df.shape[0])

X = df.apply(lambda x : x["Summary"] + " " + x["Text"], axis = 1) # nlp cleaning should be done beforehand
y = df.Score

X_train, X_test1, y_train, y_test1 = train_test_split(X, y, test_size=0.3, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test1, y_test1, test_size=0.9, random_state=42)

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