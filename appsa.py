import streamlit as st
import requests
from io import BytesIO
from PIL import Image
import numpy as np
from skimage import feature, color
import matplotlib.pyplot as plt

st.title("Simple Edge Detection from Image URL")

url = st.text_input("Enter image URL:")

if url:
    try:
        # โหลดภาพจาก URL
        response = requests.get(url)
        img = Image.open(BytesIO(response.content)).convert("RGB")

        st.image(img, caption="Original Image", use_column_width=True)

        # แปลงภาพเป็น grayscale (ขาวดำ)
        img_gray = color.rgb2gray(np.array(img))

        # ใช้ Canny edge detection
        edges = feature.canny(img_gray, sigma=2)

        # แสดงผลด้วย matplotlib แล้วแปลงเป็นภาพสำหรับ Streamlit
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(edges, cmap='gray')
        ax.set_title("Edge Detection Result")
        ax.axis('off')

        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error: {e}")
