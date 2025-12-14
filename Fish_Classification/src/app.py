import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# -------------------------------
# Load Model
# -------------------------------
MODEL_PATH = "models/mobilenet_fish.keras"

model = tf.keras.models.load_model(MODEL_PATH)

# Class names (MUST match training order)
class_names = [
    "animal fish",
    "animal fish bass",
    "fish sea_food black_sea_sprat",
    "fish sea_food gilt_head_bream",
    "fish sea_food hourse_mackerel",
    "fish sea_food red_mullet",
    "fish sea_food red_sea_bream",
    "fish sea_food sea_bass",
    "fish sea_food shrimp",
    "fish sea_food striped_red_mullet",
    "fish sea_food trout"
]

IMG_SIZE = 224

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Fish Classification", layout="centered")

st.title("🐟 Multiclass Fish Image Classification")
st.write("Upload a fish image to predict its category using a deep learning model.")

uploaded_file = st.file_uploader(
    "Choose a fish image...",
    type=["jpg", "jpeg", "png"]
)

# -------------------------------
# Image Preprocessing Function
# -------------------------------
def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# -------------------------------
# Prediction
# -------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        confidence = np.max(predictions)
        predicted_class = class_names[np.argmax(predictions)]

        st.success(f"✅ **Prediction:** {predicted_class}")
        st.info(f"📊 **Confidence:** {confidence * 100:.2f}%")
