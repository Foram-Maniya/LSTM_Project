import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import json
import os

print(f"Good evening from Surat! Starting the model training process at {
      np.datetime64('now', 's')}.")
print(f"Using TensorFlow version: {tf._version_}")

# --- Parameters ---
# These must be consistent between training and the app
NUM_WORDS = 10000  # Vocabulary size
MAX_LEN = 200      # Max review length
EMBEDDING_DIM = 128
LSTM_UNITS = 64
EPOCHS = 5
BATCH_SIZE = 64

# --- 1. Load and Preprocess Data ---
print("Loading IMDB dataset...")
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=NUM_WORDS)

print("Padding sequences...")
X_train_padded = pad_sequences(
    X_train, maxlen=MAX_LEN, padding='post', truncating='post')
X_test_padded = pad_sequences(
    X_test, maxlen=MAX_LEN, padding='post', truncating='post')

# --- 2. Build the LSTM Model ---
print("Building the LSTM model architecture...")
model = Sequential([
    Embedding(input_dim=NUM_WORDS,
              output_dim=EMBEDDING_DIM, input_length=MAX_LEN),
    LSTM(units=LSTM_UNITS),
    Dense(1, activation='sigmoid')
])

model.summary()

# --- 3. Compile the Model ---
print("Compiling the model...")
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# --- 4. Train the Model ---
print("Starting training...")
history = model.fit(
    X_train_padded,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.2
)

# --- 5. Evaluate the Model ---
loss, accuracy = model.evaluate(X_test_padded, y_test)
print(f"\nTraining complete. Test Accuracy: {accuracy*100:.2f}%")


# --- 6. Save Artifacts ---
# Save the trained model directly to the root directory
model_path = 'imdb_sentiment_lstm.h5'
print(f"Saving model to {model_path}...")
model.save(model_path)

# Save the word index (needed for encoding new reviews)
# We add 3 to the indices to account for special tokens <PAD>, <START>, <UNK>
word_index = imdb.get_word_index()
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # for unknown words

word_index_path = 'word_index.json'
print(f"Saving word index to {word_index_path}...")
with open(word_index_path, 'w') as f:
    json.dump(word_index, f)

print("\nModel and word index have been saved successfully to the root directory.")
print("You can now run the Streamlit app.")
