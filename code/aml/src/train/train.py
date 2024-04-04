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
parser.add_argument("--vector_size", type=int, help="vector size")
parser.add_argument("--window", type=int, help="window")
parser.add_argument("--min_count", type=int, help="min count")
parser.add_argument("--workers", type=int, help="workers")
parser.add_argument("--epochs", type=int, help="epochs")
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


X_train_doc2vec, X_val_doc2vec, X_test_doc2vec, doc2vec_model = doc_vectorizer(data_train, data_val, data_test, args.vectorizer,
                                                                      {'vector_size':args.vector_size, 'window':args.window, 'min_count':args.min_count, 'workers':args.workers, 'epochs':args.epochs})

model_doc2vec = build_svm(random_state=42, tol=1e-4, class_weight='balanced')

model_doc2vec, val_acc, test_acc = train_svm(model_doc2vec, X_train_doc2vec, Y_train, X_val_doc2vec, Y_val, X_test_doc2vec, Y_test)

print(val_acc, test_acc)

mlflow.log_metric("validation accuracy", val_acc)
mlflow.log_metric("test accuracy", test_acc)

# REGISTER MODEL
mlflow.sklearn.log_model(
    sk_model=model_doc2vec,
    registered_model_name=args.registered_model_name,
    artifact_path=args.registered_model_name
)

mlflow.sklearn.save_model(model_doc2vec, args.model_output)