import argparse
import pandas as pd

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential, layers
from tensorflow.keras.callbacks import EarlyStopping


import mlflow

parser = argparse.ArgumentParser()
parser.add_argument("--prep_data", type=str, help="Path of prepped data")
parser.add_argument("--registered_model_name", type=str, help="model name")
args = parser.parse_args()

df = pd.read_csv(args.prep_data, index_col="Id", encoding='utf-8')

mlflow.start_run()
mlflow.keras.autolog()


mlflow.log_metric("nb of features", df.shape[1])
mlflow.log_metric("nb of samples", df.shape[0])

X = df['Text']
y = df['Label']

# train and test split
X_train, X_test, y_train, y_test = train_test_split(df['Text'], df['Label'], test_size=0.2, random_state=42, stratify=y)


tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)  # Fit only on training data
X_train_token = tokenizer.texts_to_sequences(X_train)
X_test_token = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_token, padding='post', dtype='float32', maxlen=100)
X_test_pad = pad_sequences(X_test_token, padding='post', dtype='float32', maxlen=100)

vocab_size = len(tokenizer.word_index)
embedding_dim = 50

# Define model architecture
model = Sequential()
model.add(layers.Embedding(input_dim=vocab_size + 1,
                    output_dim=embedding_dim,
                    mask_zero=True))
model.add(layers.LSTM(20))
model.add(layers.Dense(1, activation='sigmoid'))  # Output layer for binary classification

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# print the model summary
print(model.summary())

# Train the model
es = EarlyStopping(patience=4, restore_best_weights=True)
model.fit(X_train_pad, y_train, epochs=10, batch_size=16, validation_split=0.2, callbacks=[es])


# Evaluate the model
model.evaluate(X_test_pad, y_test)

# REGISTER MODEL
mlflow.keras.log_model(
    model=model,
    registered_model_name=args.registered_model_name,
    artifact_path=args.registered_model_name
)

mlflow.end_run()