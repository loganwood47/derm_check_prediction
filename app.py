import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('models/best_model.h5')

def preprocess_image(image):
    image = Image.open(image).resize((224, 224))
    image_array = np.array(image) / 255.0
    img_array = image_array.reshape((1, 224, 224, 3))

    return(img_array)

st.title("Skin Lesion Classification")

st.write("Upload an image of a skin lesion to classify it as benign or malignant.")
st.warning("This is not a medical tool and is intended for triage only. Always consult a healthcare professional for medical advice.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
col1, col2 = st.columns(2)

if uploaded_file:
    with st.spinner("Processing image..."):
        with col1:
            st.image(uploaded_file, caption='Uploaded Image.', use_container_width=True)
        with col2:
            col2.write("Prediction:")
            img = preprocess_image(uploaded_file)
            prediction = model.predict(img)[0][0]

            if prediction > 0.5:
                st.error(f"Suspicious mole detected ({prediction:.2f} probability)! Recommend doctor follow-up.")
            else:
                st.success(f"Likely benign ({prediction:.2f} probability). Please monitor for changes and consult a physician if you are still concerned!")