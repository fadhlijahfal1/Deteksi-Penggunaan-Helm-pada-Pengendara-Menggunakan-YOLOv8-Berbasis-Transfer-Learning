import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# page config
st.set_page_config(
    page_title="Helmet Detection YOLO",
    layout="wide"
)

st.title("Helmet Detection Kelompok YOLO")

# load model
@st.cache_resource
def load_model():
    return YOLO("Weights/best.pt")

model = load_model()
classNames = ['With Helmet', 'Without Helmet']

# sidebar
source = st.sidebar.selectbox(
    "Pilih Input",
    ("Image", "Video", "Webcam")
)

conf_thres = st.sidebar.slider(
    "Confidence Threshold",
    0.1, 1.0, 0.4
)

frame_placeholder = st.empty()

# function detection
def detect_frame(frame):
    results = model(frame, conf=conf_thres, verbose=False)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = round(float(box.conf[0]), 2)

            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(
                frame,
                f"{classNames[cls]} {conf}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,0),
                2
            )
    return frame

# image mode
if source == "Image":
    uploaded_image = st.file_uploader(
        "Upload Gambar",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image:
        image = Image.open(uploaded_image)
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        frame = detect_frame(frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame, caption="Hasil Deteksi", use_container_width=True)

# video mode
elif source == "Video":
    uploaded_video = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video:
        tfile = open("temp_video.mp4", "wb")
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture("temp_video.mp4")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = detect_frame(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame_placeholder.image(
                frame,
                channels="RGB",
                use_container_width=True
            )

        cap.release()

# webcam mode
else:
    st.warning("Tekan STOP di browser untuk menghentikan webcam")

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Webcam tidak terbaca")
            break

        frame = detect_frame(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_placeholder.image(
            frame,
            channels="RGB",
            use_container_width=True
        )

    cap.release()