import streamlit as st
from PIL import Image

st.title("Deepfake Detection Tool")
st.write("Please upload an image or video to check if it's a deepfake.")

uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png", "mp4", "mov"])

if uploaded_file is not None:
    if uploaded_file.type.startswith('image'):
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)
        st.write("Image uploaded successfully!")

    elif uploaded_file.type.startswith('video'):
        st.video(uploaded_file)
        st.write("Video uploaded successfully!")
        
    if st.button("Analyse File"):
        st.write("Analysis started...")