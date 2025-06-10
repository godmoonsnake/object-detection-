import streamlit as st
from PIL import Image
import numpy as np
import tempfile
from ultralytics import YOLO

# Load the YOLOv8 model
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # or yolov8s.pt for slightly better accuracy

model = load_model()

st.title("YOLOv8 Image Detection App 🚀")
st.write("Upload an image and detect objects using YOLOv8.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Save the image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        img.save(tmp.name)
        image_path = tmp.name

    # Perform detection
    with st.spinner("Running YOLOv8 detection..."):
        results = model(image_path)
        result_img = results[0].plot()  # Draw bounding boxes

        st.image(result_img, caption="Detected Image", use_column_width=True)
        st.success("Detection complete!")

        # Show detected class names
        class_names = [model.names[int(cls)] for cls in results[0].boxes.cls]
        st.subheader("Detected Classes:")
        st.write(list(set(class_names)))
