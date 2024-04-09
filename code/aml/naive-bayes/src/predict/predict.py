import argparse

import pandas as pd

from pathlib import Path

import dill

import mlflow

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str)
parser.add_argument("--vect", type=str)
parser.add_argument("--test_data", type=str)
parser.add_argument("--predict_result", type=str)
args = parser.parse_args()

# load model and data from previous steps

data = pd.read_csv(Path(args.test_data) / "val_data.csv")
model = mlflow.sklearn.load_model(args.model)

# Use count vectorizer
with open((Path(args.vect) / "vectorizer.pkl"), 'rb') as file:
    vect = dill.load(file)

X_val_vect = vect.transform(data["CleanedText"])

data["ScorePredict"] = model.predict(X_val_vect)

data.to_csv(Path(args.predict_result) / "predict_result.csv")
