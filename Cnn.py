import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import cv2
import os

#------------------------------------------

def extract_frames(video_path, output_folder, frame_rate=1):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / frame_rate)  

  
    frame_count = 0
    saved_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Extracted {saved_count} frames from {video_path}")

video_folder = "D:\Celeb_DF"  
output_root = "D:\Celeb_DF\Frames"

for category in ["Celeb-real", "Celeb-synthesis", "YouTube-real"]:
    category_folder = os.path.join(video_folder, category)
    for video_file in os.listdir(category_folder):
        video_path = os.path.join(category_folder, video_file)
        output_folder = os.path.join(output_root, category, os.path.splitext(video_file)[0])
        extract_frames(video_path, output_folder)
        