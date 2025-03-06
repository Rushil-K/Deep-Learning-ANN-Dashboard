import streamlit as st
import pandas as pd
import numpy as np
import gdown
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

# 🎯 Google Drive File ID
file_id = "18_IlD33FyWSy1kSSEaCBfmAeyQCXqaV1"
output = "data.zip"
gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)

# 🚀 Load Dataset in Chunks
def load_data_in_chunks(file_path, chunk_size=50000):
    chunks = []
    for chunk in pd.read_csv(file_path, compression='zip', encoding='ISO-8859-1', low_memory=False, chunksize=chunk_size):
        chunks.append(chunk)
        if len(chunks) * chunk_size >= 500000:  # Limit to 500K rows for efficiency
            break
    return pd.concat(chunks, ignore_index=True)

df = load_data_in_chunks("data.zip")

# 🛠️ Preprocess Data
if 'Converted' in df.columns:
    X = df.drop(columns=['Converted'])
    y = df['Converted']
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=552627)
else:
    st.error("❌ Target column 'Converted' not found in dataset!")

# 🎨 Streamlit UI
st.title("🤖 Deep Learning ANN Dashboard")
st.sidebar.header("⚙️ Hyperparameters")

epochs = st.sidebar.slider("📅 Epochs", 1, 100, 10)
batch_size = st.sidebar.selectbox("📦 Batch Size", [16, 32, 64, 128], index=1)
learning_rate = st.sidebar.slider("⚡ Learning Rate", 0.0001, 0.1, 0.01, step=0.0001)
dropout_rate = st.sidebar.slider("🎯 Dropout Rate", 0.0, 0.5, 0.1, step=0.05)
activation_function = st.sidebar.selectbox("🔗 Activation Function", ['relu', 'sigmoid', 'tanh'], index=0)
optimizer_choice = st.sidebar.selectbox("🚀 Optimizer", ['Adam', 'SGD', 'RMSprop'], index=0)
num_layers = st.sidebar.slider("🧱 Hidden Layers", 1, 5, 3)
neurons_per_layer = st.sidebar.slider("🧠 Neurons per Layer", 8, 256, 64, step=8)

# 📌 Build ANN Model
def create_model():
    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(X_train.shape[1],)))
    for _ in range(min(num_layers, 3)):  # Limit to 3 layers for efficiency
        model.add(keras.layers.Dense(min(neurons_per_layer, 64), activation=activation_function))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(dropout_rate))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    if optimizer_choice == "Adam":
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_choice == "SGD":
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

model = create_model()

# 🎯 Train Model Button
if st.button("🚀 Train Model"):
    with st.spinner("Training in progress... ⏳"):
        history = model.fit(X_train, y_train, epochs=min(epochs, 10), batch_size=batch_size, validation_data=(X_test, y_test), verbose=1)
        
        # 📌 Display Model Accuracy
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        st.success(f"✅ Final Train Accuracy: {final_train_acc:.4f} | Final Validation Accuracy: {final_val_acc:.4f}")

        # 📊 Plot Performance
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].plot(history.history['loss'], label='Train Loss', color='red')
        ax[0].plot(history.history['val_loss'], label='Validation Loss', color='blue')
        ax[0].legend()
        ax[0].set_title("📉 Loss Curve")

        ax[1].plot(history.history['accuracy'], label='Train Accuracy', color='green')
        ax[1].plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
        ax[1].legend()
        ax[1].set_title("📈 Accuracy Curve")

        st.pyplot(fig)

        # 🎯 ROC Curve
        y_pred_proba = model.predict(X_test).ravel()
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('📊 ROC Curve')
        ax.legend(loc="lower right")
        st.pyplot(fig)

# 🎨 Additional Creative Visuals
st.subheader("📊 Data Insights")

# 🔘 Pairplot for Feature Relations (Only using first 5 features for efficiency)
st.subheader("📊 Feature Relations (Pairplot)")
pairplot_features = df.iloc[:, :5]  # Limit to 5 columns for faster rendering
sns.pairplot(pairplot_features)
st.pyplot(plt)

# 🔘 Pie Chart of Target Variable
if 'Converted' in df.columns:
    fig, ax = plt.subplots()
    df['Converted'].value_counts().plot.pie(autopct='%1.1f%%', colors=['#ff9999', '#66b3ff'], ax=ax, wedgeprops={'edgecolor': 'black'})
    ax.set_title("🔘 Converted Distribution")
    st.pyplot(fig)

# 📊 Horizontal Bar Chart of Feature Averages
st.subheader("📊 Feature Averages")
feature_means = df.iloc[:, :10].mean().sort_values()
fig, ax = plt.subplots()
feature_means.plot(kind='barh', color=sns.color_palette("coolwarm", len(feature_means)), ax=ax)
ax.set_title("📊 Top 10 Feature Averages")
st.pyplot(fig)
