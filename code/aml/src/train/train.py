import argparse

import pandas as pd

from gensim.corpora import Dictionary
from gensim.models import LdaModel

import mlflow

parser = argparse.ArgumentParser()
parser.add_argument("--training_data", type=str, help="Path of prepped data")
parser.add_argument("--registered_model_name", type=str, help="model name")
parser.add_argument("--nb_passes", type=int, help="nb passes")
args = parser.parse_args()

df = pd.read_csv(args.training_data)

mlflow.start_run()
mlflow.sklearn.autolog()

print(df.head())
print(df.shape)

mlflow.log_metric("nb of features", df.shape[1])
mlflow.log_metric("nb of samples", df.shape[0])

df.dropna(inplace=True)

# Convert the 'tokens' column into a list of lists for Gensim
documents = df['CleanedText'].tolist()

# Create a Gensim Dictionary
dictionary = Dictionary(documents)

# Filter extremes to refine the dictionary
dictionary.filter_extremes(no_below=5, no_above=0.5)

# Create a Corpus
corpus = [dictionary.doc2bow(doc) for doc in documents]

# Train the LDA model
lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=2, passes=10)

# Explore the topics
for idx, topic in lda_model.print_topics(-1):
    print(f"Topic: {idx} \nWords: {topic}")

mlflow.save_model(lda_model, args.model_output)