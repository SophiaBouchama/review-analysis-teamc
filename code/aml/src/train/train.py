import argparse

import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import MiniBatchKMeans

from utils import get_clusters_info

import mlflow

parser = argparse.ArgumentParser()
parser.add_argument("--training_data", type=str, help="Path of prepped data")
parser.add_argument("--registered_model_name", type=str, help="model name")
parser.add_argument("--init_size", type=int, help="init size")
parser.add_argument("--batch_size", type=int, help="bactch size")
parser.add_argument("--n_init", type=float, help="n init")
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

# KMEANS
num_clusters = 5
kmeans_model = MiniBatchKMeans(n_clusters=num_clusters, init='k-means++', n_init=args.n_init, 
                         init_size=args.init_size, batch_size=args.batch_size, verbose=False, max_iter=1000, random_state=42)
kmeans = kmeans_model.fit(X_train_vect)

# for training
print("train clusters")
get_clusters_info(kmeans, X_train_vect, vect, num_clusters)

# for val
print("val clusters")
get_clusters_info(kmeans, X_val_vect, vect, num_clusters)

# for test
print("test clusters")
get_clusters_info(kmeans, X_test_vect, vect, num_clusters)

mlflow.log_metric("model_inertia", kmeans.inertia_)

# REGISTER MODEL
mlflow.sklearn.log_model(
    sk_model=kmeans,
    registered_model_name=args.registered_model_name,
    artifact_path=args.registered_model_name
)

mlflow.sklearn.save_model(kmeans, args.model_output)
