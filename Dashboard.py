import streamlit as st
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

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
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)

    # Display Accuracy
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
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
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

    # 📊 Prediction Distribution
    st.subheader("📊 Distribution of Predictions")
    fig_pred, ax_pred = plt.subplots()
    sns.histplot(y_pred_prob, bins=30, kde=True, color="purple")
    ax_pred.set_xlabel("Predicted Probability")
    ax_pred.set_ylabel("Frequency")
    st.pyplot(fig_pred)

    # 📊 Feature Importance (SHAP)
    st.subheader("🔍 Feature Importance (SHAP)")
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)
    fig_shap, ax_shap = plt.subplots()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    st.pyplot(fig_shap)

    # 📊 Feature Correlation
    st.subheader("📊 Feature Correlation Heatmap")
    fig_corr, ax_corr = plt.subplots(figsize=(6, 4))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax_corr)
    st.pyplot(fig_corr)

    # 📊 Data Distribution Before Training (Pairplot)
    st.subheader("📊 Data Distribution Before Training")
    sample_df = df.sample(min(1000, len(df)))  # Adjust sample size for efficiency
    fig_pair = sns.pairplot(sample_df, diag_kind="kde")
    st.pyplot(fig_pair)
