import matplotlib.pyplot as plt
import cv2
import os
import shutil
import random
from concurrent.futures import ThreadPoolExecutor

#----------------------------------------------------

def extract_media(input_path, output_folder, frame_rate=1, max_frames=10):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    saved_count = 0

    if input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            print(f"Error: Could not open video {input_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps / frame_rate)

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret or saved_count >= max_frames:
                break

            if frame_count % frame_interval == 0:
                frame_filename = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
                cv2.imwrite(frame_filename, frame)
                saved_count += 1

            frame_count += 1

        cap.release()

    elif input_path.lower().endswith(('.jpg', '.jpeg', '.png')):
        img = cv2.imread(input_path)
        if img is not None:
            frame_filename = os.path.join(output_folder, "image_0000.jpg")
            cv2.imwrite(frame_filename, img)
            saved_count = 1

    if saved_count == 0:
        print(f"No frames extracted from: {input_path}")
        shutil.rmtree(output_folder, ignore_errors=True)
    else:
        print(f"Extracted {saved_count} frames from {input_path}")

# ----------------------------------------------------

video_folder = r"D:\Celeb_DF"
output_root = r"D:\Celeb_DF\Frames"
categories = ["Celeb-real", "Celeb-synthesis", "YouTube-real"]

def extract_all():
    tasks = []

    with ThreadPoolExecutor(max_workers=8) as executor:
        for category in categories:
            category_folder = os.path.join(video_folder, category)

            if not os.path.exists(category_folder):
                continue

            for media_file in os.listdir(category_folder):
                media_path = os.path.join(category_folder, media_file)
                media_name = os.path.splitext(media_file)[0]
                output_folder = os.path.join(output_root, category, media_name)

                tasks.append(executor.submit(extract_media, media_path, output_folder, 1, 10))

# ----------------------------------------------------

def split_data(source, train_dir, validation_dir, test_dir, train_ratio=0.7, validation_ratio=0.15, max_images=20):
    for folder in [train_dir, validation_dir, test_dir]:
        os.makedirs(os.path.join(folder, "real"), exist_ok=True)
        os.makedirs(os.path.join(folder, "fake"), exist_ok=True)

    counts = {
        "real": {"train": 0, "validation": 0, "test": 0},
        "fake": {"train": 0, "validation": 0, "test": 0}
    }

    real_folders = []
    for real_cat in ["Celeb-real", "YouTube-real"]:
        real_root = os.path.join(source, real_cat)
        if not os.path.exists(real_root):
            continue
        for subfolder in os.listdir(real_root):
            subpath = os.path.join(real_root, subfolder)
            if os.path.isdir(subpath):
                real_folders.append((subfolder, subpath))
    real_count = len(real_folders)
    print(f"Found {real_count} real sample folders.")

    fake_folders = []
    fake_root = os.path.join(source, "Celeb-synthesis")
    for subfolder in os.listdir(fake_root):
        subpath = os.path.join(fake_root, subfolder)
        if os.path.isdir(subpath):
            fake_folders.append((subfolder, subpath))
    print(f"Found {len(fake_folders)} fake sample folders.")

    if len(fake_folders) > real_count:
        print(f"Downsampling fake folders from {len(fake_folders)} to {real_count}")
        random.shuffle(fake_folders)
        fake_folders = fake_folders[:real_count]

    combined = [("real", real_folders), ("fake", fake_folders)]

    for label, folders in combined:
        for subfolder_name, subfolder_path in folders:
            files = [f for f in os.listdir(subfolder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if not files:
                continue
            random.shuffle(files)
            files = files[:max_images]

            train_split = int(len(files) * train_ratio)
            validation_split = train_split + int(len(files) * validation_ratio)

            for i, item in enumerate(files):
                source_path = os.path.join(subfolder_path, item)

                if i < train_split:
                    split = "train"
                    dest = os.path.join(train_dir, label, subfolder_name, item)
                elif i < validation_split:
                    split = "validation"
                    dest = os.path.join(validation_dir, label, subfolder_name, item)
                else:
                    split = "test"
                    dest = os.path.join(test_dir, label, subfolder_name, item)

                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copy2(source_path, dest)
                counts[label][split] += 1

    print("\nDataset Split Summary:")
    for label in counts:
        total = sum(counts[label].values())
        print(f"  {label.upper()} — Total: {total}")
        for split in ["train", "validation", "test"]:
            print(f"    {split.capitalize()}: {counts[label][split]}")

if __name__ == "__main__":
    extract_all()

    source = r"D:\Celeb_DF\Frames"
    train_dir = r"D:\Celeb_DF\Frames\train"
    validation_dir = r"D:\Celeb_DF\Frames\validation"
    test_dir = r"D:\Celeb_DF\Frames\test"

    split_data(source, train_dir, validation_dir, test_dir)