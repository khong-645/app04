import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import cv2
from ultralytics import YOLO
import numpy as np

st.title("Object Detection from Image URL using YOLO")

url = st.text_input("Enter image URL:")

if url:
    try:
        # โหลดภาพจาก URL
        response = requests.get(url)
        img_pil = Image.open(BytesIO(response.content)).convert("RGB")
        
        st.image(img_pil, caption="Original Image", use_column_width=True)
        
        # แปลง PIL Image เป็น numpy array (BGR) สำหรับ OpenCV
        img = np.array(img_pil)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # โหลดโมเดล YOLOv8 รุ่นเล็ก (yolov8n.pt)
        model = YOLO('yolov8n.pt')
        
        # ตรวจจับวัตถุ
        results = model(img)
        
        # วาดกรอบและ label ลงบนภาพ (ผลลัพธ์เป็น numpy array)
        result_img = results[0].plot()
        
        # แปลงกลับเป็น RGB สำหรับแสดงใน Streamlit
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        
        st.image(result_img, caption="Detected Objects", use_column_width=True)
        
        # แสดงข้อมูลวัตถุที่ตรวจจับได้
        st.write("Detected Objects:")
        for box in results[0].boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = box
            label = model.names[int(class_id)]
            st.write(f"- {label} (Confidence: {score:.2f})")

    except Exception as e:
        st.error(f"Error: {e}")
