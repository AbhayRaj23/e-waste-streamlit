import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from utils.preprocess import prepare_image

class_names = ['Battery', 'Keyboard', 'Microwave', 'Mobile', 'Mouse', 
               'PCB', 'Player', 'Printer', 'Television', 'Washing Machine']

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/efficientnet_model.h5")

model = load_model()

st.title("🔍 E-Waste Image Classifier")
st.write("Upload an image to classify it into one of 10 e-waste categories.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = prepare_image(image)
    prediction = model.predict(img_array)
    pred_label = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"**Prediction:** {pred_label} ({confidence:.2f}%)")
