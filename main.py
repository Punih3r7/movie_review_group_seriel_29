# Bangla Movie Review Sentiment Classification (RNN, 3 variants)
# --------------------------------------------------------------
# This notebook-style pipeline is designed to support your final report:
#
# Sections it helps with:
# - Dataset: load, EDA, cleaning
# - Methodology: model architecture, hyperparameters, fine-tuning details
# - Training procedure: losses, optimizers, seeds, environment
# - Results: metrics, tables, figures, confusion matrices
# - Discussion: limitations and ethical considerations (printed as a draft)
#
# It also:
# - Trains 3 RNN variants: underfitting, "good" fit, overfitting
# - Saves many plots into ./figures/ (you can download and insert in LaTeX)
# - Saves best model + tokenizer to the current directory

import os
import random
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pickle

with open("training_history.pkl", "wb") as f:
    pickle.dump(results, f)


from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from IPython.display import display

# ------------------------------
# 0. Reproducibility + environment
# ------------------------------

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("===== Environment Info =====")
print("Working directory:", os.getcwd())
print("Directory contents:", os.listdir())
print("TensorFlow version:", tf.__version__)
print("Available GPUs:", tf.config.list_physical_devices("GPU"))
print("Random seed:", SEED)
print("============================\n")

# ------------------------------
# 1. Load the Excel file
# ------------------------------

path = "movie_reviews_dataset_2000_bangla.xlsx"
if not os.path.exists(path):
    raise FileNotFoundError(f"File not found at {path} - please confirm upload.")

df = pd.read_excel(path)
print("\nLoaded dataframe shape:", df.shape)
print("\nColumns and dtypes:")
print(df.dtypes)

print("\nFirst 10 raw rows:")
display(df.head(10))

# ------------------------------
# 2. Identify text and label columns (heuristics)
# ------------------------------

text_col = None
label_col = None

# Text column: object dtype, name contains review/text/comment/sentence/remarks
for col in df.columns:
    if df[col].dtype == object:
        name = str(col).lower()
        if any(k in name for k in ["review", "text", "comment", "sentence", "remarks"]):
            text_col = col
            break

# Fallback: first object column
if text_col is None:
    for col in df.columns:
        if df[col].dtype == object:
            text_col = col
            break

# Label column: label/sentiment/rating/score/target/class
for col in df.columns:
    name = str(col).lower()
    if any(k in name for k in ["label", "sentiment", "rating", "score", "target", "class"]):
        label_col = col
        break

# Fallback: numeric column with few unique values
if label_col is None:
    numeric_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    for c in numeric_cols:
        if 1 < df[c].nunique() <= 10:
            label_col = c
            break

if text_col is None:
    raise ValueError("Couldn't detect a text column automatically. Please specify it.")
if label_col is None:
    raise ValueError("Couldn't detect a label column automatically. Please specify it.")

print(f"\nDetected text column : {text_col}")
print(f"Detected label column: {label_col}")

# ------------------------------
# 3. Cleaning pipeline for Bangla text
# ------------------------------

def clean_text_bangla(s):
    if pd.isna(s):
        return ""
    s = str(s)
    s = s.strip()
    # Remove URLs and emails
    s = re.sub(r"http\S+|www\S+|https\S+", " ", s)
    s = re.sub(r"\S+@\S+", " ", s)
    # Remove English letters and digits
    s = re.sub(r"[A-Za-z0-9]", " ", s)
    # Keep Bengali range and basic punctuation, remove other symbols
    s = re.sub(r"[^\u0980-\u09FF\s।?!-]", " ", s)
    # Normalize whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s

df["_clean_text"] = df[text_col].apply(clean_text_bangla)

print("\nSample of raw vs cleaned text:")
display(df[[text_col, "_clean_text", label_col]].head(12))

# ------------------------------
# 4. Prepare labels (binary sentiment)
# ------------------------------

y_raw = df[label_col].copy()

# String labels: try mapping to positive (1) / negative (0)
if y_raw.dtype == object:
    y_lower = y_raw.astype(str).str.lower()
    mapping = {}
    for val in y_lower.unique():
        if any(k in val for k in ["pos", "positive", "ভালো", "good"]):
            mapping[val] = 1
        elif any(k in val for k in ["neg", "negative", "খারাপ", "bad"]):
            mapping[val] = 0
    if mapping:
        y = y_lower.map(mapping)
    else:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(y_lower)
else:
    # Numeric: if more than 2 classes, convert (>= mid -> 1, else 0)
    if y_raw.nunique() > 2:
        mid = (float(y_raw.max()) + float(y_raw.min())) / 2.0
        y = (y_raw.astype(float) >= mid).astype(int)
    else:
        y = y_raw.astype(int)

print("\nLabel distribution after processing:")
print(pd.Series(y).value_counts())

df["_label"] = y

# Drop empty cleaned texts or missing labels
initial_len = len(df)
df = df[(df["_clean_text"].str.len() > 0) & df["_label"].notna()]
print(f"\nDropped {initial_len - len(df)} rows with empty text or missing labels. Remaining rows: {len(df)}")

# ------------------------------
# 5. EDA: sentiment distribution & text length
# ------------------------------

os.makedirs("figures", exist_ok=True)

print("\n===== EDA: Sentiment Distribution =====")
sent_counts = df["_label"].value_counts().sort_index()
print(sent_counts)

plt.figure()
sent_counts.plot(kind="bar")
plt.xticks([0, 1], ["Negative (0)", "Positive (1)"], rotation=0)
plt.title("Sentiment Label Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("figures/sentiment_distribution.png")
plt.show()

# Text length in characters
char_lengths = df["_clean_text"].astype(str).apply(len)
print("\n===== Text Length (characters) =====")
print(char_lengths.describe())

plt.figure()
plt.hist(char_lengths, bins=30)
plt.title("Review Text Length Distribution (characters)")
plt.xlabel("Length (characters)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("figures/text_length_distribution.png")
plt.show()

# ------------------------------
# 6. Tokenization and sequence preparation
# ------------------------------

texts = df["_clean_text"].tolist()
labels = df["_label"].astype(int).tolist()

vocab_size = 10000
maxlen = 100

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=maxlen, padding="post", truncating="post")
y = np.array(labels)

print("\nVocabulary size (limit) :", vocab_size)
print("Max sequence length      :", maxlen)
print("Padded tensor shape      :", X.shape)
print("Sample tokenized sequence:", sequences[0][:30])

# Token length distribution (before padding)
token_lengths = [len(seq) for seq in sequences]
print("\n===== Token Length (before padding) =====")
print(pd.Series(token_lengths).describe())

plt.figure()
plt.hist(token_lengths, bins=30)
plt.title("Review Length Distribution (tokens)")
plt.xlabel("Length (tokens)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("figures/token_length_distribution.png")
plt.show()

# Train/val/test split with stratification
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.25, random_state=SEED, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp
)

print("\nTrain/Val/Test sizes:", X_train.shape[0], X_val.shape[0], X_test.shape[0])

# ------------------------------
# 7. Build RNN models (3 variants)
# ------------------------------

def build_rnn_model(vocab_size, embed_dim, rnn_units, dropout_rate=0.0):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=maxlen))
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))
    model.add(SimpleRNN(rnn_units, return_sequences=False))
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

results = {}
models = {}
eval_summary = {}

callbacks_common = [
    EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
]

# Underfitting model (very small)
print("\n===== Training UNDERFITTING model =====")
under_model = build_rnn_model(vocab_size=vocab_size, embed_dim=16, rnn_units=8, dropout_rate=0.0)
under_model.build(input_shape=(None, maxlen))
print(under_model.summary())
history_under = under_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10, batch_size=64,
    verbose=2,
    callbacks=callbacks_common
)
models["underfit"] = under_model
results["underfit"] = history_under.history

# Good model
print("\n===== Training GOOD model =====")
good_model = build_rnn_model(vocab_size=vocab_size, embed_dim=64, rnn_units=64, dropout_rate=0.3)
good_model.build(input_shape=(None, maxlen))
print(good_model.summary())

# Use native Keras format (.keras) to avoid HDF5 legacy warning
checkpoint_path = "best_rnn_model.keras"
mc = ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True, save_weights_only=False)

history_good = good_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20, batch_size=32,
    verbose=2,
    callbacks=callbacks_common + [mc]
)
models["good"] = good_model
results["good"] = history_good.history

# Overfitting model (large capacity, no dropout)
print("\n===== Training OVERFITTING model =====")
over_model = build_rnn_model(vocab_size=vocab_size, embed_dim=200, rnn_units=256, dropout_rate=0.0)
over_model.build(input_shape=(None, maxlen))
print(over_model.summary())
history_over = over_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20, batch_size=32,
    verbose=2,
    callbacks=callbacks_common
)
models["overfit"] = over_model
results["overfit"] = history_over.history

# ------------------------------
# 8. Plot training curves (many figures)
# ------------------------------

def plot_history(hist, name):
    os.makedirs("figures", exist_ok=True)

    # Loss
    plt.figure(figsize=(8, 4))
    plt.plot(hist["loss"], label="train_loss")
    plt.plot(hist["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{name} model - Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/{name}_loss.png")
    plt.show()

    # Accuracy
    plt.figure(figsize=(8, 4))
    plt.plot(hist["accuracy"], label="train_acc")
    plt.plot(hist["val_accuracy"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{name} model - Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/{name}_accuracy.png")
    plt.show()

plot_history(results["underfit"], "underfit")
plot_history(results["good"], "good")
plot_history(results["overfit"], "overfit")

# ------------------------------
# 9. Evaluate on test set + confusion matrices
# ------------------------------

def plot_confusion_matrix(cm, name):
    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(f"Confusion Matrix - {name} model")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Negative", "Positive"])
    plt.yticks(tick_marks, ["Negative", "Positive"])
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black"
            )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(f"figures/confusion_matrix_{name}.png")
    plt.show()

for name, model in models.items():
    print(f"\n=== Evaluation for {name.upper()} model ===")
    y_pred_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_pred_prob >= 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    eval_summary[name] = {"accuracy": acc, "f1": f1}
    print("Accuracy:", acc)
    print("F1-score:", f1)
    print("Classification report:")
    print(classification_report(y_test, y_pred, digits=4))
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:\n", cm)
    plot_confusion_matrix(cm, name)

# ------------------------------
# 10. Summary table of models
# ------------------------------

summary = pd.DataFrame({
    "model": ["underfit", "good", "overfit"],
    "train_loss_last": [results[k]["loss"][-1] for k in ["underfit", "good", "overfit"]],
    "val_loss_last": [results[k]["val_loss"][-1] for k in ["underfit", "good", "overfit"]],
    "train_acc_last": [results[k]["accuracy"][-1] for k in ["underfit", "good", "overfit"]],
    "val_acc_last": [results[k]["val_accuracy"][-1] for k in ["underfit", "good", "overfit"]],
    "test_accuracy": [eval_summary[k]["accuracy"] for k in ["underfit", "good", "overfit"]],
    "test_f1": [eval_summary[k]["f1"] for k in ["underfit", "good", "overfit"]],
})
print("\n===== Model comparison summary =====")
display(summary)

# ------------------------------
# 11. Save tokenizer and good model
# ------------------------------

tokenizer_path = "tokenizer.json"
with open(tokenizer_path, "w", encoding="utf-8") as f:
    f.write(tokenizer.to_json())

saved_files = []
if os.path.exists(checkpoint_path):
    saved_files.append(checkpoint_path)
if os.path.exists(tokenizer_path):
    saved_files.append(tokenizer_path)

print("\nSaved files in current directory:")
for fpath in saved_files:
    print(" -", fpath)

# ------------------------------
# 12. Print REPORT SECTIONS (for direct copy to LaTeX)
# ------------------------------

acc_good = eval_summary["good"]["accuracy"]
f1_good = eval_summary["good"]["f1"]


