import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np

# Load models (ensure the paths are correct)
chest_model = load_model('Dpl/chest_disease_detection_model.h5')
tb_model = load_model('Dpl/tb_disease_detection_model.h5')
eye_model = load_model('Dpl/retinal_disease_detection_model.h5')


# eye_model = load_model('eye_model.h5')


# Function to preprocess the image
def preprocess_image(image, target_size=(128,128)):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


# Function to predict
def predict(image, model):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return prediction


# Streamlit app
st.title("Medical Image Analysis")

# User selects the area
area = st.selectbox("Select the area of concern:", ["Chest","Eye"])

# User uploads an image
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Analyze"):
        if area == "Chest":
            # Run chest and TB models
            chest_prediction = predict(image, chest_model)
            tb_prediction = predict(image, tb_model)
            st.write("Chest Model Prediction:", "Healthy" if chest_prediction[0][0] > 0.5 else "Not Healthy")
            st.write("TB Model Prediction:", "Healthy" if tb_prediction[0][0] > 0.5 else "Not Healthy")

        elif area == "Eye":
            # Run eye model
            eye_prediction = predict(image, eye_model)
            st.write("Eye Model Prediction:", "Healthy" if eye_prediction[0][0] > 0.5 else "Not Healthy")

        # elif area == "Skin":
        #     # Run skin model
        #     skin_prediction = predict(image, skin_model)
        #     st.write("Skin Model Prediction:", "Healthy" if skin_prediction[0][0] > 0.5 else "Not Healthy")
