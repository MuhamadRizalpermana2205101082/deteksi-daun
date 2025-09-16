import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.title("🌿 Deteksi Daun Tanaman Hias")

# Load model
model = tf.keras.models.load_model("model_daun.h5")

uploaded_file = st.file_uploader("Upload gambar daun", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((128, 128))  # sesuaikan dengan input model
    st.image(image, caption="Gambar diupload", use_column_width=True)

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    kelas = np.argmax(pred, axis=1)[0]

    st.write(f"Hasil Prediksi: **{kelas}**")
