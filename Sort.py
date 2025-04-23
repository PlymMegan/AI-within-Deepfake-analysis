import cv2
import os
import shutil
import numpy as np
import random
from tqdm import tqdm

#----------------------------------------------------

def extract_frames(video_dir, output_dir, num_frames=10, image_size=(224, 224)):
    os.makedirs(output_dir, exist_ok=True)

    for category in ["Celeb-real", "Celeb-synthesis", "YouTube-real"]:
        category_path = os.path.join(video_dir, category)
        if not os.path.exists(category_path):
            continue

        for video_name in tqdm(os.listdir(category_path), desc=f"Processing {category}"):
            if not video_name.endswith(('.mp4', '.avi', '.mov')):
                continue

            video_path = os.path.join(category_path, video_name)
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames < num_frames:
                print(f"Skipping {video_name} (only {total_frames} frames)")
                continue

            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            saved_frames = 0
            frame_count = 0

            video_base = os.path.splitext(video_name)[0]
            video_out_dir = os.path.join(output_dir, category, video_base)
            os.makedirs(video_out_dir, exist_ok=True)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count in frame_indices:
                    frame = cv2.resize(frame, image_size)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    frame_filename = os.path.join(video_out_dir, f"frame_{saved_frames:03d}.jpg")
                    cv2.imwrite(frame_filename, frame)
                    saved_frames += 1

                frame_count += 1

            cap.release()


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

    fake_folders = []
    fake_root = os.path.join(source, "Celeb-synthesis")
    for subfolder in os.listdir(fake_root):
        subpath = os.path.join(fake_root, subfolder)
        if os.path.isdir(subpath):
            fake_folders.append((subfolder, subpath))

    if len(fake_folders) > len(real_folders):
        print(f"Downsampling fake folders from {len(fake_folders)} to {len(real_folders)}")
        random.shuffle(fake_folders)
        fake_folders = fake_folders[:len(real_folders)]

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
    video_folder = r"D:\Celeb_DF"
    output_root = r"D:\Celeb_DF\Frames"
    extract_frames(video_folder, output_root)

    source = r"D:\Celeb_DF\Frames"
    train_dir = r"D:\Celeb_DF\Frames\train"
    validation_dir = r"D:\Celeb_DF\Frames\validation"
    test_dir = r"D:\Celeb_DF\Frames\test"

    split_data(source, train_dir, validation_dir, test_dir)
