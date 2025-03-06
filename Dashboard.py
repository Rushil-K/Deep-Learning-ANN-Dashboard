import streamlit as st
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import gdown
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

# Set Streamlit page config
st.set_page_config(page_title="ANN Dashboard", layout="wide")

# Title
st.title("🔬 ANN Dashboard for Classification")

# Load CSV from Google Drive
drive_url = "https://drive.google.com/uc?id=18_IlD33FyWSy1kSSEaCBfmAeyQCXqaV1"
file_path = "dataset.csv"
gdown.download(drive_url, file_path, quiet=False)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(file_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="ISO-8859-1", errors="replace")
    return df

df = load_data()

if df.empty:
    st.error("❌ Failed to load dataset! Check the file link or encoding.")
    st.stop()

st.subheader("📊 Data Preview")
st.write(df.head())

# Sidebar Hyperparameter Selection
st.sidebar.header("⚙️ Model Hyperparameters")
test_size = st.sidebar.slider("🧪 Test Size", 0.1, 0.5, 0.2)
epochs = st.sidebar.slider("⏳ Epochs", 5, 100, 10)
batch_size = st.sidebar.selectbox("📦 Batch Size", [16, 32, 64, 128], index=1)
neurons_layer1 = st.sidebar.slider("🔢 Layer 1 Neurons", 16, 128, 64)
neurons_layer2 = st.sidebar.slider("🔢 Layer 2 Neurons", 16, 128, 32)
dropout_rate = st.sidebar.slider("💧 Dropout Rate", 0.0, 0.5, 0.2)
activation_function = st.sidebar.selectbox("⚡ Activation", ["relu", "tanh", "sigmoid"], index=0)
optimizer = st.sidebar.selectbox("🚀 Optimizer", ["adam", "sgd", "rmsprop"], index=0)

# Check if target column exists
target_column = "Converted"
if target_column not in df.columns:
    st.error(f"⚠️ Error: Target column '{target_column}' not found!")
    st.stop()

X = pd.get_dummies(df.drop(columns=[target_column]))
y = df[target_column]
X.fillna(X.mean(), inplace=True)
y.fillna(y.mode()[0], inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=552627)

# Build ANN Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(neurons_layer1, activation=activation_function, input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(dropout_rate),
    tf.keras.layers.Dense(neurons_layer2, activation=activation_function),
    tf.keras.layers.Dropout(dropout_rate),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

if st.button("🚀 Train Model"):
    with st.spinner("Training in Progress..."):
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    st.success(f"✅ Test Accuracy: {accuracy:.4f}")

    # Visualization
    st.subheader("📈 Training Performance")
    fig, ax = plt.subplots()
    ax.plot(history.history['accuracy'], label="Training Accuracy", color="blue")
    ax.plot(history.history['loss'], label="Training Loss", color="red")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Score")
    ax.legend()
    st.pyplot(fig)

    # Confusion Matrix
    st.subheader("📊 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", cbar=False)
    st.pyplot(fig)

    # ROC Curve
    st.subheader("📈 ROC Curve & AUC Score")
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig)

st.success("🎯 Dashboard Loaded Successfully!")
