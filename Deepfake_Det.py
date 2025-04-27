# User Interface (Streamlit)

import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import tempfile
import cv2
from tensorflow.keras.models import load_model

# ------------------------------------------

model = load_model('best_model.keras')

st.title("Deepfake Detection Tool")
st.write("Please upload an image or video to check if it's a deepfake.")

uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png", "mp4", "mov"])

def extract_frames(video_path, num_frames=10):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    idx = 0
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i == frame_indices[idx]:
            frame = cv2.resize(frame, (224, 224))  # Resize to 224x224
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)
            idx += 1
            if idx >= len(frame_indices):
                break
    cap.release()
    return np.array(frames)

if uploaded_file is not None:
    if uploaded_file.type.startswith('image'):
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_container_width=True)
        st.write("Image uploaded successfully!")

    elif uploaded_file.type.startswith('video'):
        st.video(uploaded_file)
        st.write("Video uploaded successfully!")

    if st.button("Analyse File"):
        st.spinner("Analyzing file...")

        if uploaded_file.type.startswith('image'):
            image = Image.open(uploaded_file).convert('RGB')
            image = image.resize((224, 224))
            image = np.array(image) / 255.0
            image = np.expand_dims(image, axis=0)  # (1, 224, 224, 3)
            image = np.expand_dims(image, axis=0)  # (1, 1, 224, 224, 3) for TimeDistributed

            prediction = model.predict(image)
            result = "Deepfake" if prediction[0] > 0.5 else "Real"

        elif uploaded_file.type.startswith('video'):
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())

            frames = extract_frames(tfile.name)

            if frames.shape[0] < 10:
                st.error("Video too short to analyze properly.")
                result = None
            else:
                frames = np.expand_dims(frames, axis=0)  # (1, 10, 224, 224, 3)
                prediction = model.predict(frames)
                result = "Deepfake" if prediction[0] > 0.5 else "Real"

        if result:
            st.success(f"Result: {result}")