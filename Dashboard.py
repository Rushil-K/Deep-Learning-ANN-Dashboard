import streamlit as st
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 📌 Title
st.title("🔬 ANN Dashboard for Classification")

# 📤 Load the fixed CSV file
csv_path = "nmrk2627_df_processed.csv"
df = pd.read_csv(csv_path)

# Sidebar: Hyperparameters
st.sidebar.header("⚙️ Model Hyperparameters")

# Define target and feature columns
target_column = "Converted"  # Assuming 'Converted' is the target
feature_columns = [col for col in df.columns if col != target_column]

# Split data
X = df[feature_columns]
y = df[target_column]
test_size = st.sidebar.slider("🧪 Test Set Ratio", 0.1, 0.5, 0.2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# 🔧 Hyperparameter Controls
epochs = st.sidebar.slider("⏳ Epochs", 5, 100, 10)
batch_size = st.sidebar.selectbox("📦 Batch Size", [16, 32, 64, 128], index=1)
neurons_layer1 = st.sidebar.slider("🔢 Neurons in Layer 1", 16, 128, 64)
neurons_layer2 = st.sidebar.slider("🔢 Neurons in Layer 2", 16, 128, 32)
dropout_rate = st.sidebar.slider("💧 Dropout Rate", 0.0, 0.5, 0.2)
activation_function = st.sidebar.selectbox("⚡ Activation Function", ["relu", "tanh", "sigmoid"], index=0)

# ANN Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(neurons_layer1, activation=activation_function, input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(dropout_rate),
    tf.keras.layers.Dense(neurons_layer2, activation=activation_function),
    tf.keras.layers.Dropout(dropout_rate),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile Model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train Model Button
if st.button("🚀 Train Model"):
    with st.spinner("Training in Progress..."):
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    # Evaluate Model
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)

    # Display Accuracy
    st.success(f"✅ Test Accuracy: {accuracy:.4f}")

    # 📊 Training Performance Plot
    st.subheader("📈 Training Performance")
    fig, ax = plt.subplots()
    ax.plot(history.history['accuracy'], label="Training Accuracy")
    ax.plot(history.history['loss'], label="Training Loss")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Score")
    ax.legend()
    st.pyplot(fig)

    # 📊 Heatmap for Correlation
    st.subheader("📊 Feature Correlation")
    fig_corr, ax_corr = plt.subplots(figsize=(6, 4))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax_corr)
    st.pyplot(fig_corr)

