import matplotlib.pyplot as plt
import cv2
import os
import shutil
import random

#----------------------------------------------------

def extract_media(input_path, output_folder, frame_rate=1):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            print(f"Error: Could not open video {input_path}")
            return

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
        print(f"Extracted {saved_count} frames from {input_path}")

    elif input_path.lower().endswith(('.jpg', '.jpeg', '.png')):
        img = cv2.imread(input_path)
        if img is not None:
            frame_filename = os.path.join(output_folder, "image_0000.jpg")
            cv2.imwrite(frame_filename, img)
            print(f"Copied image to {frame_filename}")


video_folder = r"D:\Celeb_DF"
output_root = r"D:\Celeb_DF\Frames"

for category in ["Celeb-real", "Celeb-synthesis", "YouTube-real"]:
    category_folder = os.path.join(video_folder, category)

    for media_file in os.listdir(category_folder):
        media_path = os.path.join(category_folder, media_file)
        media_name = os.path.splitext(media_file)[0]
        output_folder = os.path.join(output_root, category, media_name)

        extract_media(media_path, output_folder)

def split_data(source, train_dir, validation_dir, test_dir, train_ratio=0.7, validation_ratio=0.15):
    for folder in [train_dir, validation_dir, test_dir]:
        os.makedirs(os.path.join(folder, "real"), exist_ok=True)
        os.makedirs(os.path.join(folder, "fake"), exist_ok=True)

    for category in ["Celeb-real", "YouTube-real", "Celeb-synthesis"]:
        label = "real" if "real" in category else "fake"
        category_path = os.path.join(source, category)

        subcategories = os.listdir(category_path)

        for subcategory in subcategories:
            subcategory_path = os.path.join(category_path, subcategory)

            videos_or_images = os.listdir(subcategory_path)
            random.shuffle(videos_or_images)

            train_split = int(len(videos_or_images) * train_ratio)
            validation_split = train_split + int(len(videos_or_images) * validation_ratio)

            for i, item in enumerate(videos_or_images):
                source_path = os.path.join(subcategory_path, item)

                if i < train_split:
                    dest = os.path.join(train_dir, label, subcategory, item)
                elif i < validation_split:
                    dest = os.path.join(validation_dir, label, subcategory, item)
                else:
                    dest = os.path.join(test_dir, label, subcategory, item)

                if os.path.isdir(source_path):
                    shutil.copytree(source_path, dest)
                else:
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    shutil.copy2(source_path, dest)



source = r"D:\Celeb_DF\Frames"
train_dir = r"D:\Celeb_DF\Frames\train"
validation_dir = r"D:\Celeb_DF\Frames\validation"
test_dir = r"D:\Celeb_DF\Frames\test"

split_data(source, train_dir, validation_dir, test_dir)
