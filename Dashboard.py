import streamlit as st
import pandas as pd
import numpy as np
import gdown
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

# 🎯 Download dataset from Google Drive
file_id = "18_IlD33FyWSy1kSSEaCBfmAeyQCXqaV1"
output = "data.zip"
gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)

# 📊 Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("data.zip", compression='zip', encoding='ISO-8859-1', low_memory=False)
    return df

df = load_data()

# 🔄 Preprocess data
if 'Converted' in df.columns:
    X = df.drop(columns=['Converted'])
    y = df['Converted']
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=552627)
else:
    st.error("⚠️ Target column 'Converted' not found in dataset!")

# 🎨 Streamlit UI with aesthetics
st.title("🚀 Deep Learning ANN Dashboard")
st.markdown("### 📊 Model Training & Performance Visualization")

st.sidebar.header("🎛️ Hyperparameters")
epochs = st.sidebar.slider("⏳ Epochs", 1, 100, 10)
batch_size = st.sidebar.selectbox("📦 Batch Size", [16, 32, 64, 128], index=1)
learning_rate = st.sidebar.slider("🚀 Learning Rate", 0.0001, 0.1, 0.01, step=0.0001)
dropout_rate = st.sidebar.slider("🎭 Dropout Rate", 0.0, 0.5, 0.1, step=0.05)
activation_function = st.sidebar.selectbox("⚡ Activation Function", ['relu', 'sigmoid', 'tanh'], index=0)
num_layers = st.sidebar.slider("🏗️ Number of Hidden Layers", 1, 5, 3)
neurons_per_layer = st.sidebar.slider("🧠 Neurons per Layer", 8, 256, 64, step=8)
optimizer_choice = st.sidebar.selectbox("⚙️ Optimizer", ["adam", "sgd", "rmsprop"], index=0)

# 🔧 Build ANN model
def create_model():
    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(X_train.shape[1],)))
    for _ in range(num_layers):
        model.add(keras.layers.Dense(neurons_per_layer, activation=activation_function))
        model.add(keras.layers.Dropout(dropout_rate))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    
    optimizer = {
        "adam": keras.optimizers.Adam(learning_rate=learning_rate),
        "sgd": keras.optimizers.SGD(learning_rate=learning_rate),
        "rmsprop": keras.optimizers.RMSprop(learning_rate=learning_rate)
    }[optimizer_choice]
    
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

model = create_model()
st.write("📝 **Model Summary:**")
st.text(model.summary())

# 🚀 Train model
with st.spinner("⏳ Training the model... Please wait!"):
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=(X_test, y_test))

# 📈 Plot performance
st.markdown("## 📊 Training Performance")

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(history.history['loss'], label='Train Loss', color='red', linestyle='dashed', marker='o')
ax[0].plot(history.history['val_loss'], label='Validation Loss', color='blue', marker='o')
ax[0].legend()
ax[0].set_title("📉 Loss Curve")
ax[0].grid()

ax[1].plot(history.history['accuracy'], label='Train Accuracy', color='green', linestyle='dashed', marker='o')
ax[1].plot(history.history['val_accuracy'], label='Validation Accuracy', color='purple', marker='o')
ax[1].legend()
ax[1].set_title("📈 Accuracy Curve")
ax[1].grid()

st.pyplot(fig)

st.success("✅ Training Completed Successfully!")
