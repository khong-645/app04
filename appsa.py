import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
from ultralytics import YOLO
import numpy as np

st.title("Object Detection from Image URL using YOLO (No cv2)")

url = st.text_input("Enter image URL:")

if url:
    try:
        # โหลดภาพจาก URL
        response = requests.get(url)
        img_pil = Image.open(BytesIO(response.content)).convert("RGB")
        
        st.image(img_pil, caption="Original Image", use_column_width=True)
        
        # แปลง PIL Image เป็น numpy array (RGB)
        img_np = np.array(img_pil)

        # โหลดโมเดล YOLOv8 รุ่นเล็ก (yolov8n.pt)
        model = YOLO('yolov8n.pt')
        
        # ตรวจจับวัตถุ
        results = model(img_np)
        
        # ใช้ PIL วาดกรอบและ label ลงบนภาพ
        img_draw = img_pil.copy()
        draw = ImageDraw.Draw(img_draw)
        
        # โหลดฟอนต์ (อันนี้ใช้ default)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        for box in results[0].boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = box
            label = model.names[int(class_id)]
            conf_text = f"{label} {score:.2f}"
            
            # วาดกรอบ
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            
            # วาดข้อความ label
            text_size = draw.textsize(conf_text, font=font)
            draw.rectangle([x1, y1 - text_size[1], x1 + text_size[0], y1], fill="red")
            draw.text((x1, y1 - text_size[1]), conf_text, fill="white", font=font)
        
        st.image(img_draw, caption="Detected Objects", use_column_width=True)
        
        # แสดงข้อมูลวัตถุที่ตรวจจับได้
        st.write("Detected Objects:")
        for box in results[0].boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = box
            label = model.names[int(class_id)]
            st.write(f"- {label} (Confidence: {score:.2f})")

    except Exception as e:
        st.error(f"Error: {e}")
