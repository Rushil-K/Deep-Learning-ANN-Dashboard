import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
import os
import gdown

# 📌 Streamlit Title
st.title("🔬 ANN Dashboard for Classification")

# 📤 Google Drive CSV Download
file_id = "18_IlD33FyWSy1kSSEaCBfmAeyQCXqaV1"
csv_filename = "data.csv"

@st.cache_data
def load_data():
    if not os.path.exists(csv_filename):
        gdown.download(f"https://drive.google.com/uc?id={file_id}", csv_filename, quiet=False)
    
    try:
        df = pd.read_csv(csv_filename, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_filename, encoding="ISO-8859-1")

    return df

df = load_data()

# ✅ Show dataset preview
st.write("### 📂 Dataset Preview", df.head())

# 🚨 Validate target column exists
target_column = "Converted"  
if target_column not in df.columns:
    st.error(f"⚠️ Error: Target column '{target_column}' not found! Available columns: {list(df.columns)}")
    st.stop()

# Feature columns (excluding target)
feature_columns = [col for col in df.columns if col != target_column]

# 🚀 Handle missing values
df.dropna(inplace=True)

# 🚀 Convert categorical columns to numeric (if any)
df = pd.get_dummies(df, columns=df.select_dtypes(include=['object']).columns, drop_first=True)

# Split dataset
X = df[feature_columns].values  # Ensure NumPy array format
y = df[target_column].values  # Ensure NumPy array format

test_size = st.sidebar.slider("🧪 Test Set Ratio", 0.1, 0.5, 0.2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=552627)

# 🔧 Model Hyperparameters
st.sidebar.header("⚙️ Model Hyperparameters")
epochs = st.sidebar.slider("⏳ Epochs", 5, 100, 10)
batch_size = st.sidebar.selectbox("📦 Batch Size", [16, 32, 64, 128], index=1)
neurons_layer1 = st.sidebar.slider("🔢 Neurons in Layer 1", 16, 128, 64)
neurons_layer2 = st.sidebar.slider("🔢 Neurons in Layer 2", 16, 128, 32)
dropout_rate = st.sidebar.slider("💧 Dropout Rate", 0.0, 0.5, 0.2)
activation_function = st.sidebar.selectbox("⚡ Activation Function", ["relu", "tanh", "sigmoid"], index=0)
optimizer = st.sidebar.selectbox("🚀 Optimizer", ["adam", "sgd", "rmsprop"], index=0)

# 🏗️ Build ANN Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(neurons_layer1, activation=activation_function, input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(dropout_rate),
    tf.keras.layers.Dense(neurons_layer2, activation=activation_function),
    tf.keras.layers.Dropout(dropout_rate),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile Model
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

# 🚀 Train Model Button
if st.button("🚀 Train Model"):
    with st.spinner("Training in Progress..."):
        history = model.fit(
            np.array(X_train), 
            np.array(y_train), 
            epochs=epochs, 
            batch_size=batch_size, 
            verbose=0
        )

    # Evaluate Model
    y_pred_prob = model.predict(np.array(X_test))
    y_pred = (y_pred_prob > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)

    st.success(f"✅ Test Accuracy: {accuracy:.4f}")

    # 📈 Plot Training Performance
    st.subheader("📈 Loss vs Accuracy Over Epochs")
    fig_hist, ax_hist = plt.subplots()
    ax_hist.plot(history.history['accuracy'], label="Training Accuracy", color="blue")
    ax_hist.plot(history.history['loss'], label="Training Loss", color="red")
    ax_hist.set_xlabel("Epochs")
    ax_hist.set_ylabel("Score")
    ax_hist.legend()
    st.pyplot(fig_hist)
