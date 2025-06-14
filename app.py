import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="Wafer Pass/Fail Detector", layout="centered")
st.markdown("<h1 style='text-align: center; color: #5A5A5A;'>🧠 Semiconductor Wafer Pass/Fail Detector</h1>", unsafe_allow_html=True)

st.markdown("### Upload Wafer Image:")
uploaded_file = st.file_uploader("Choose a 30x30 grayscale wafer image (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    img_array = np.array(image.resize((30, 30)))
    st.image(image, caption="Uploaded Wafer", width=200)

    model = load_model('wafer_model.h5')
    input_img = img_array.reshape(1, 30, 30, 1) / 255.0
    prediction = model.predict(input_img)[0][0]

    if prediction < 0.5:
        st.success("✅ Result: PASS")
    else:
        st.error("❌ Result: FAIL")
    st.progress(min(int(prediction * 100), 100))