import tensorflow as tf
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import requests
from io import BytesIO
from streamlit_option_menu import option_menu

st.set_page_config(page_title="Butterfly Image Classification", layout="centered")

# Load model
model = load_model('butterflyresnet50.hdf5')

# Load class names
try:
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        'train',
        shuffle=True,
        image_size=(224, 224),
        batch_size=32
    )
    class_names = dataset.class_names
except Exception as e:
    st.warning("Could not load class names from train/ folder. Using hardcoded list.")
    class_names = ['Monarch', 'Swallowtail', 'Blue Morpho', 'Painted Lady', 'Common Buckeye', 'Unknown']



# Prediction function
def predict_image(img):
    img = img.resize((224, 224))
    img_array = tf.expand_dims(tf.keras.preprocessing.image.img_to_array(img), 0)
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    predicted_index = np.argmax(score)

    if predicted_index < len(class_names):
        return class_names[predicted_index], score[predicted_index].numpy()
    else:
        return "Unknown Butterfly", 0.0

# Image loader from URL
def load_image_from_url(url):
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        return img
    except Exception as e:
        st.error(f"Error loading image from URL: {e}")
        return None

# UI
st.title("🦋 Butterfly Image Classification")

selected = option_menu(
    menu_title=None,
    options=["Upload Image", "Image by URL"],
    icons=["upload", "link"],
    default_index=0,
    orientation="horizontal"
)

if selected == "Upload Image":
    uploaded_file = st.file_uploader("Upload a butterfly image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)
        if st.button("Predict"):
            label, confidence = predict_image(img)
            st.success(f"Predicted: {label} ({confidence*100:.2f}% confidence)")

if selected == "Image by URL":
    url = st.text_input("Enter image URL (must be direct image link)")
    if url:
        img = load_image_from_url(url)
        if img:
            st.image(img, caption="Image from URL", use_column_width=True)
            if st.button("Predict"):
                label, confidence = predict_image(img)
                st.success(f"Predicted: {label} ({confidence*100:.2f}% confidence)")
