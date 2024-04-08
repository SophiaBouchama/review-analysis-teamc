import argparse
import dill
from dill import dump

import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

import mlflow

parser = argparse.ArgumentParser()
parser.add_argument("--training_data_folder", type=str, help="Folder path of prepped data")
parser.add_argument("--training_data_name", type=str, help="Name of the prepped data")
parser.add_argument("--train_data", type=str, help="Folder path to train data folder")

args = parser.parse_args()

df = pd.read_csv((Path(args.training_data_folder) / args.training_data_name))

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
vect = CountVectorizer().fit(X_train)

# pickle vectorizer 
with open((Path(args.train_data) / "vectorizer.pkl"), 'wb') as file:
    dill.dump(vect, file)






