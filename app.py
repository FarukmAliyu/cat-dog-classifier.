import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# --- Configuration ---
MODEL_PATH = 'cat_dog_model.h5'
IMAGE_SIZE = (128, 128)
CLASS_NAMES = ['Cat', 'Dog'] 
# Cat is class 0, Dog is class 1, based on the default alphabetical order of ImageDataGenerator

@st.cache_resource 
def load_model():
    """Loads the pre-trained Keras model and caches it."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Error: Model file not found at {MODEL_PATH}.")
        st.stop()
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

def preprocess_image(image):
    """Resizes and normalizes the image for model prediction."""
    image = image.resize(IMAGE_SIZE)
    # Convert image to numpy array and normalize
    img_array = np.array(image) / 255.0
    # Add batch dimension (1, 128, 128, 3)
    img_array = np.expand_dims(img_array, axis=0) 
    return img_array

def make_prediction(model, processed_image):
    """Runs the model prediction."""
    prediction = model.predict(processed_image)
    # Sigmoid output is a probability between 0 and 1
    probability = prediction[0][0]
    
    # Classify: probability < 0.5 is Cat (0), >= 0.5 is Dog (1)
    if probability >= 0.5:
        predicted_class = CLASS_NAMES[1] # Dog
        confidence = probability
    else:
        predicted_class = CLASS_NAMES[0] # Cat
        confidence = 1 - probability
        
    return predicted_class, confidence

# --- Streamlit App Interface ---
def main():
    st.set_page_config(page_title="Cat vs. Dog Classifier", layout="centered")
    st.title("🐾 Cat vs. Dog Image Classifier")
    st.markdown("Upload an image and the model will predict if it's a **Cat** or a **Dog**.")

    # 1. Load the Model
    model = load_model()

    # 2. File Uploader Widget
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        try:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            st.write("")
            st.markdown("### Classifying...")

            # 3. Preprocess and Predict
            processed_image = preprocess_image(image)
            
            # The spinner provides visual feedback during inference
            with st.spinner("Model is running inference..."):
                predicted_class, confidence = make_prediction(model, processed_image)

            # 4. Display Results
            if predicted_class == 'Dog':
                st.balloons()
                st.success(f"Prediction: **{predicted_class}**")
            else:
                st.success(f"Prediction: **{predicted_class}**")

            st.write(f"Confidence: **{confidence:.2f}**")
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

if __name__ == "__main__":
    main()