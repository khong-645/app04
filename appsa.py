import streamlit as st
from PIL import Image
import requests
from io import BytesIO
from ultralytics import YOLO

st.title("Object Detection from Image URL")

# รับ URL รูปภาพจากผู้ใช้
image_url = st.text_input("Enter Image URL:")

if image_url:
    try:
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        
        st.image(img, caption="Original Image", use_column_width=True)
        
        # โหลดโมเดล YOLOv8 (หรือ YOLOv5 หากใช้เวอร์ชันนั้น)
        model = YOLO('yolov8n.pt')  # รุ่นเล็กเพื่อความเร็ว
        
        # ตรวจจับวัตถุในภาพ
        results = model(img)
        
        # แสดงภาพที่มีกรอบวัตถุ
        result_img = results[0].plot()  # plot() คืนค่าเป็น numpy array
        
        st.image(result_img, caption="Detected Objects", use_column_width=True)
        
        # แสดงข้อมูลวัตถุที่ตรวจจับได้
        st.write("Detected Objects:")
        for obj in results[0].boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = obj
            label = model.names[int(class_id)]
            st.write(f"- {label} (Confidence: {score:.2f})")
            
    except Exception as e:
        st.error(f"Error loading or processing image: {e}")
