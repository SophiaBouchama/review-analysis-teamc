import argparse

import pandas as pd

from sklearn.model_selection import train_test_split

import mlflow

from utils import doc_vectorizer, build_svm, train_svm

parser = argparse.ArgumentParser()
parser.add_argument("--training_data", type=str, help="Path of prepped data")
parser.add_argument("--registered_model_name", type=str, help="model name")
parser.add_argument("--test_data", type=str, help="Path to test data")
parser.add_argument("--model_output", type=str, help="Path of output model")
parser.add_argument("--vectorizer", type=str, help="vectorizer")
parser.add_argument("--min_df", type=str, help="min df")
parser.add_argument("--ngram_range_min", type=str, help="ngram min")
parser.add_argument("--ngram_range_max", type=str, help="ngram max")
parser.add_argument("--max_features", type=str, help="max features")
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

data_train, data_val_test, Y_train, Y_val_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=X['Score']
)

data_val, data_test, Y_val, Y_test = train_test_split(
    data_val_test, Y_val_test, test_size=0.333, random_state=42, stratify=data_val_test['Score']
)

data_train = data_train['CleanedText']
data_val = data_val['CleanedText']
data_test = data_test['CleanedText']


X_train_tfidf, X_val_tfidf, X_test_tfidf = doc_vectorizer(data_train, data_val, data_test,
args.vectorizer, {'min_df':args.min_df, 'ngram_range':(args.ngram_range_min, args.ngram_range_max)})

model_tfidf = build_svm(random_state=42, tol=1e-4, class_weight='balanced')

model_tfidf, val_acc, test_acc = train_svm(model_tfidf, X_train_tfidf, Y_train, X_val_tfidf, Y_val, X_test_tfidf, Y_test)

print(val_acc, test_acc)

# REGISTER MODEL
mlflow.sklearn.log_model(
    sk_model=model_tfidf,
    registered_model_name=args.registered_model_name,
    artifact_path=args.registered_model_name
)

mlflow.sklearn.save_model(model_tfidf, args.model_output)
