import streamlit as st
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import requests
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier

# 📌 Title
st.title("🔬 ANN Dashboard for Classification")

# 📤 Load the CSV file from Google Drive
csv_url = "https://drive.google.com/uc?id=18_IlD33FyWSy1kSSEaCBfmAeyQCXqaV1"

@st.cache_data
def load_data(url):
    response = requests.get(url)
    response.raise_for_status()  # Ensure successful response
    df = pd.read_csv(BytesIO(response.content), encoding="utf-8")  # Handle encoding issues
    return df

# Try loading the data
try:
    df = load_data(csv_url)
except Exception as e:
    st.error(f"❌ Error loading CSV file: {e}")
    st.stop()

# Display dataset preview
st.subheader("📄 Dataset Preview")
st.write(df.head())

# Display column names
st.write("🔍 Available Columns:", list(df.columns))

# Ensure the target column exists
target_column = "Converted"  # Adjust if necessary
if target_column not in df.columns:
    st.error(f"⚠️ Error: Target column '{target_column}' not found! Please check available columns above.")
    st.stop()  # Stop execution if column is missing

# Define feature columns
feature_columns = [col for col in df.columns if col != target_column]

# Split data
X = df[feature_columns]
y = df[target_column]
test_size = st.sidebar.slider("🧪 Test Set Ratio", 0.1, 0.5, 0.2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=552627)

# Sidebar: Hyperparameters
st.sidebar.header("⚙️ Model Hyperparameters")
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

    st.success(f"✅ Test Accuracy: {accuracy:.4f}")

    # 📊 Training Performance Plot
    st.subheader("📈 Loss vs Accuracy Over Epochs")
    fig_hist, ax_hist = plt.subplots()
    ax_hist.plot(history.history['accuracy'], label="Training Accuracy", color="blue")
    ax_hist.plot(history.history['loss'], label="Training Loss", color="red")
    ax_hist.set_xlabel("Epochs")
    ax_hist.set_ylabel("Score")
    ax_hist.legend()
    st.pyplot(fig_hist)

    # 📊 Confusion Matrix
    st.subheader("📊 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", cbar=False, 
                xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
    ax_cm.set_xlabel("Predicted Label")
    ax_cm.set_ylabel("True Label")
    ax_cm.set_title("Confusion Matrix")
    st.pyplot(fig_cm)

    # 📊 ROC Curve
    st.subheader("📈 ROC Curve & AUC Score")
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, color="blue", label=f"AUC = {roc_auc:.2f}")
    ax_roc.plot([0, 1], [0, 1], color="gray", linestyle="--")  # Diagonal reference line
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend(loc="lower right")
    st.pyplot(fig_roc)

    # 📊 Feature Importance Using RandomForest
    st.subheader("📊 Feature Importance (RandomForest Surrogate)")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=552627)
    rf_model.fit(X_train, y_train)

    feature_importance = rf_model.feature_importances_
    sorted_idx = np.argsort(feature_importance)

    fig_feat, ax_feat = plt.subplots(figsize=(8, 5))
    ax_feat.barh(np.array(feature_columns)[sorted_idx], feature_importance[sorted_idx], color="orange")
    ax_feat.set_xlabel("Importance Score")
    ax_feat.set_title("Feature Importance (RandomForest)")
    st.pyplot(fig_feat)

    # 📊 Class Distribution Pie Chart
    st.subheader("📊 Class Distribution")
    fig_pie, ax_pie = plt.subplots()
    labels = ["Not Converted", "Converted"]
    counts = [sum(y_train == 0), sum(y_train == 1)]
    ax_pie.pie(counts, labels=labels, autopct="%1.1f%%", colors=["red", "green"], startangle=90)
    st.pyplot(fig_pie)

    # 📊 Data Distribution Before Training
    st.subheader("📊 Data Distribution Before Training")
    sample_df = df.sample(min(1000, len(df)))  # Adjust sample size for efficiency
    fig_pair = sns.pairplot(sample_df, diag_kind="kde")
    st.pyplot(fig_pair)
