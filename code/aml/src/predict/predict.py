import argparse
import pandas as pd
import mlflow

mlflow.sklearn.autolog()

parser = argparse.ArgumentParser("predict")
parser.add_argument("--model_input", type=str, help="Path of input model")
parser.add_argument("--test_data", type=str, help="Path to test data")
parser.add_argument("--predictions", type=str, help="Path of predictions")

args = parser.parse_args()

print("hello predict world...")

lines = [
    f"Model path: {args.model_input}",
    f"Test data path: {args.test_data}",
    f"Predictions path: {args.predictions}",
]

print(lines)

# Load test data
df = pd.read_csv(args.test_data)
testX = df['CleanedText']

# Load the model from input port
model = mlflow.sklearn.load_model(args.model_input)

# Make predictions on testX data and record them in a column named predicted_cost
predictions = model.predict(testX)
testX["predicted_sentiment"] = predictions
print(testX.shape)

# Compare predictions to actuals (df.Score)
output_data = pd.DataFrame(testX)
output_data["actual_sentiment"] = df["Score"]


# Save the output data with feature columns, predicted cost, and actual cost in csv file
output_data = output_data.to_csv(args.predictions)