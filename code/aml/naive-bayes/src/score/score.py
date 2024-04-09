import argparse

import pandas as pd

from pathlib import Path

import mlflow

from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str)
parser.add_argument("--predict_result", type=str)
parser.add_argument("--score_report", type=str)
args = parser.parse_args()

data = pd.read_csv(Path(args.predict_result) / "predict_result.csv")

model = mlflow.sklearn.load_model(args.model)

actual = data["Score"]
predictions = data["ScorePredict"]

# accuracy score
print(f"Accuracy Score: {accuracy_score(actual, predictions)}")
print(f"Model: {model}")

# Print score report to a text file
(Path(args.score_report) / "score.txt").write_text(f"Scored with the following model:{model}\n")

with open((Path(args.score_report) / "score.txt"), "a") as f:
    f.write(f"Accuracy: {accuracy_score(actual, predictions)} \n")