# app.py
import os
import streamlit as st
import gdown
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# -------------------------------
# Step 1: Model Download Setup
# -------------------------------

# Replace YOUR_FILE_ID_HERE with the ID from your Google Drive link
MODEL_FILE_ID = "YOUR_FILE_ID_HERE"
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
MODEL_PATH = "cat_dog_model.h5"

# Download the model if it doesn't exist
if not os.path.exists(MODEL_PATH):
    st.write("Downloading model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    st.write("Model downloaded successfully!")

# Load the model
model = tf.keras.models.load_model(MODEL_PATH)

# -------------------------------
# Step 2: Streamlit App UI
# -------------------------------

st.title("Cat & Dog Classifier 🐱🐶")
st.write("Upload an image of a cat or dog, and the model will predict it!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image for your model
    img = img.resize((224, 224))  # adjust if your model uses a different size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # shape (1, 224, 224, 3)
    img_array = img_array / 255.0  # normalize if your model expects it

    # Predict
    prediction = model.predict(img_array)
    
    # Assuming binary classification: 0=cat, 1=dog
    label = "Dog 🐶" if prediction[0][0] > 0.5 else "Cat 🐱"
    st.write(f"Prediction: **{label}**")
