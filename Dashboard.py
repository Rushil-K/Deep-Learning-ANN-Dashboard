import streamlit as st
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import gdown
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier

# Title
st.title("🔬 ANN Dashboard for Classification")

# Load the CSV file from Google Drive
csv_url = "https://drive.google.com/uc?id=18_IlD33FyWSy1kSSEaCBfmAeyQCXqaV1"
@st.cache_data
def load_data():
    file_path = "data.csv"
    gdown.download(csv_url, file_path, quiet=False)
    df = pd.read_csv(file_path)
    return df

df = load_data()

# Ensure dataset integrity
st.write("### Sample Data")
st.write(df.head())
st.write("### Data Info")
st.write(df.info())

# Sidebar: Hyperparameters
st.sidebar.header("⚙️ Model Hyperparameters")

target_column = "Converted"  # Update if different
if target_column not in df.columns:
    st.error(f"⚠️ Error: Target column '{target_column}' not found! Check dataset columns.")
    st.stop()

feature_columns = [col for col in df.columns if col != target_column]
X = df[feature_columns]
y = df[target_column]

# Convert categorical variables if any
X = pd.get_dummies(X)

# Split data
random_state = 552627
test_size = st.sidebar.slider("🧪 Test Set Ratio", 0.1, 0.5, 0.2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# Convert to NumPy arrays to avoid training errors
X_train = np.array(X_train).astype(np.float32)
X_test = np.array(X_test).astype(np.float32)
y_train = np.array(y_train).astype(np.float32)
y_test = np.array(y_test).astype(np.float32)

# Hyperparameter Controls
epochs = st.sidebar.slider("⏳ Epochs", 5, 100, 10)
batch_size = st.sidebar.selectbox("📦 Batch Size", [16, 32, 64, 128], index=1)
neurons_layer1 = st.sidebar.slider("🔢 Neurons in Layer 1", 16, 128, 64)
neurons_layer2 = st.sidebar.slider("🔢 Neurons in Layer 2", 16, 128, 32)
dropout_rate = st.sidebar.slider("💧 Dropout Rate", 0.0, 0.5, 0.2)
activation_function = st.sidebar.selectbox("⚡ Activation Function", ["relu", "tanh", "sigmoid"], index=0)
optimizer = st.sidebar.selectbox("🚀 Optimizer", ["adam", "sgd", "rmsprop"], index=0)

# ANN Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(neurons_layer1, activation=activation_function, input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(dropout_rate),
    tf.keras.layers.Dense(neurons_layer2, activation=activation_function),
    tf.keras.layers.Dropout(dropout_rate),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile Model
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

# Train Model Button
if st.button("🚀 Train Model"):
    with st.spinner("Training in Progress..."):
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    # Evaluate Model
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)

    # Display Accuracy
    st.success(f"✅ Test Accuracy: {accuracy:.4f}")

    # Training Performance Plot
    st.subheader("📈 Loss vs Accuracy Over Epochs")
    fig_hist, ax_hist = plt.subplots()
    ax_hist.plot(history.history['accuracy'], label="Training Accuracy", color="blue")
    ax_hist.plot(history.history['loss'], label="Training Loss", color="red")
    ax_hist.set_xlabel("Epochs")
    ax_hist.set_ylabel("Score")
    ax_hist.legend()
    st.pyplot(fig_hist)

    # Confusion Matrix
    st.subheader("📊 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", cbar=False, xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
    ax_cm.set_xlabel("Predicted Label")
    ax_cm.set_ylabel("True Label")
    ax_cm.set_title("Confusion Matrix")
    st.pyplot(fig_cm)

    # ROC Curve
    st.subheader("📈 ROC Curve & AUC Score")
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, color="blue", label=f"AUC = {roc_auc:.2f}")
    ax_roc.plot([0, 1], [0, 1], color="gray", linestyle="--")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend(loc="lower right")
    st.pyplot(fig_roc)

    # Feature Importance
    st.subheader("📊 Feature Importance (RandomForest Surrogate)")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    rf_model.fit(X_train, y_train)
    feature_importance = rf_model.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    fig_feat, ax_feat = plt.subplots(figsize=(8, 5))
    ax_feat.barh(np.array(feature_columns)[sorted_idx], feature_importance[sorted_idx], color="orange")
    ax_feat.set_xlabel("Importance Score")
    ax_feat.set_title("Feature Importance (RandomForest)")
    st.pyplot(fig_feat)

    # Class Distribution
    st.subheader("📊 Class Distribution")
    fig_pie, ax_pie = plt.subplots()
    labels = ["Not Converted", "Converted"]
    counts = [sum(y_train == 0), sum(y_train == 1)]
    ax_pie.pie(counts, labels=labels, autopct="%1.1f%%", colors=["red", "green"], startangle=90)
    st.pyplot(fig_pie)
