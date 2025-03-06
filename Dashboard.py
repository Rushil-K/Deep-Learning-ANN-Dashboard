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

# 🎯 Google Drive File ID
file_id = "18_IlD33FyWSy1kSSEaCBfmAeyQCXqaV1"
output = "data.zip"
gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)

# 🚀 Load Dataset in Chunks
def load_data_in_chunks(file_path, chunk_size=50000):
    chunks = []
    for chunk in pd.read_csv(file_path, compression='zip', encoding='ISO-8859-1', low_memory=False, chunksize=chunk_size):
        chunks.append(chunk)
        if len(chunks) * chunk_size >= 500000:  # Limit to 500K rows for faster training
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

# 🚀 Optimize Data Loading with tf.data
BUFFER_SIZE = 10000
BATCH_SIZE = 32  # Default batch size
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

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
        history = model.fit(train_dataset, epochs=min(epochs, 10), validation_data=test_dataset, verbose=1)

        # 📊 Plot Performance
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].plot(history.history['loss'], label='Train Loss')
        ax[0].plot(history.history['val_loss'], label='Validation Loss')
        ax[0].legend()
        ax[0].set_title("📉 Loss Curve")

        ax[1].plot(history.history['accuracy'], label='Train Accuracy')
        ax[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax[1].legend()
        ax[1].set_title("📈 Accuracy Curve")

        st.pyplot(fig)

# 🎨 Additional Creative Visuals
st.subheader("📊 Data Insights")

# 🔘 Pie Chart of Target Variable
if 'Converted' in df.columns:
    fig, ax = plt.subplots()
    df['Converted'].value_counts().plot.pie(autopct='%1.1f%%', colors=['#ff9999', '#66b3ff'], ax=ax)
    ax.set_title("🔘 Converted Distribution")
    st.pyplot(fig)

# 📊 Horizontal Bar Chart of First 10 Features
st.subheader("📊 Feature Distributions")
feature_means = df.iloc[:, :10].mean().sort_values()
fig, ax = plt.subplots()
feature_means.plot(kind='barh', color='skyblue', ax=ax)
ax.set_title("📊 Top 10 Feature Averages")
st.pyplot(fig)

# ✅ GPU Check
gpu_available = len(tf.config.experimental.list_physical_devices('GPU')) > 0
st.sidebar.write(f"⚡ GPU Available: {'✅ Yes' if gpu_available else '❌ No'}")
