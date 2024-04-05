import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential, layers
from tensorflow.keras.callbacks import EarlyStopping
import mlflow

parser = argparse.ArgumentParser()
parser.add_argument("--training_data", type=str, help="Path of prepped data")
parser.add_argument("--registered_model_name", type=str, help="model name")
parser.add_argument("--lstm_units_layer", type=int, help="lstm units layer")
parser.add_argument("--embedding_dim", type=int, help="embedding dim")
parser.add_argument("--epoch", type=int, help="embedding dim")
parser.add_argument("--model_output", type=str, help="Path of output model")
parser.add_argument("--test_data", type=str, help="Path to test data")
args = parser.parse_args()

# Load data
df = pd.read_csv(args.training_data, index_col="Id", encoding='utf-8')

mlflow.start_run()
mlflow.keras.autolog()

# Log number of features and samples
mlflow.log_metric("nb of features", df.shape[1])
mlflow.log_metric("nb of samples", df.shape[0])

X = df['Text']
y = to_categorical(df['Score'] - 1)

# Train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# tokenize and pad
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
X_train_token = tokenizer.texts_to_sequences(X_train)
X_test_token = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_token, padding='post', dtype='float32', maxlen=100)
X_test_pad = pad_sequences(X_test_token, padding='post', dtype='float32', maxlen=100)

vocab_size = len(tokenizer.word_index) + 1
embedding_dim = args.embedding_dim

# Define model architecture for multiclass classification
model = Sequential([
    layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True),
    layers.LSTM(10),
    layers.Dense(5, activation='softmax')  # this time with 5 outputs
])

# multiclass classification compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# printing model summary
print(model.summary())

# Train the model
es = EarlyStopping(patience=4, restore_best_weights=True)
model.fit(X_train_pad, y_train, epochs=args.epoch, batch_size=16, validation_split=0.2, callbacks=[es])

# Evaluate the model
model.evaluate(X_test_pad, y_test)

# Register model with MLflow
mlflow.keras.log_model(
    model=model,
    registered_model_name=args.registered_model_name,
    artifact_path=args.registered_model_name
)

mlflow.sklearn.save_model(model, args.model_output)
