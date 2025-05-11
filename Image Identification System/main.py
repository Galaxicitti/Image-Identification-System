import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)
from PIL import Image


@st.cache_resource
def load_model():
    try:
        model = MobileNetV2(weights="imagenet")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None


def preprocess_image(image):
    img = np.array(image.convert("RGB"))  # Ensure 3 channels
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img


def classify_image(model, image):
    try:
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        decoded_predictions = decode_predictions(predictions, top=5)[0]
        return decoded_predictions
    except Exception as e:
        st.error(f"Error classifying image: {str(e)}")
        return None


def main():
    st.set_page_config(page_title="Image Identification System", page_icon="🖼️", layout="centered")
    
    st.title("🖼️ Image Identification System")
    st.write("Upload an image and let AI tell you what's in it!")

    model = load_model()
    
    if model is None:
        st.error("Model could not be loaded. Please try again later.")
        return
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp", "tiff"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=700)  # Adjust width as needed


        with st.spinner("Analyzing Image..."):
            predictions = classify_image(model, image)
        
        if predictions:
            st.subheader("Predictions (Top 5)")
            for i, (imagenet_id, label, score) in enumerate(predictions, start=1):
                st.write(f"**{i}. {label.title()}** ({imagenet_id}): {score:.2%}")
        else:
            st.warning("No valid predictions were found.")
    else:
        st.info("Please upload an image to get started.")


if __name__ == "__main__":
    main()
