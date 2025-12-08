
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

MODEL_PATH = "cat_dog_model.h5"

model = tf.keras.models.load_model(MODEL_PATH)

st.title("Cats vs Dogs Classifier")
st.write("Upload an image of a cat or dog")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).resize((224,224))
    st.image(image, caption="Uploaded Image")

    img_array = np.array(image)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        st.success("✅ It's a DOG")
    else:
        st.success("✅ It's a CAT")
