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

    def collect_folders(categories):
        folders = []
        for cat in categories:
            root = os.path.join(source, cat)
            if not os.path.exists(root):
                continue
            for subfolder in os.listdir(root):
                subpath = os.path.join(root, subfolder)
                if os.path.isdir(subpath):
                    folders.append((subfolder, subpath))
        return folders

    real_folders = collect_folders(["Celeb-real", "YouTube-real"])
    fake_folders = collect_folders(["Celeb-synthesis"])

    print(f"Found {len(real_folders)} real sample folders.")
    print(f"Found {len(fake_folders)} fake sample folders.")

    random.shuffle(real_folders)
    random.shuffle(fake_folders)

    def split_balanced(folders, label):
        num_total = len(folders)
        num_val = min(len(real_folders), len(fake_folders)) // 5
        num_test = num_val
        num_train = num_total - num_val - num_test

        return {
            "train": folders[:num_train],
            "validation": folders[num_train:num_train + num_val],
            "test": folders[num_train + num_val:]
        }

    real_split = split_balanced(real_folders, "real")
    fake_split = split_balanced(fake_folders, "fake")

    counts = {
        "real": {"train": 0, "validation": 0, "test": 0},
        "fake": {"train": 0, "validation": 0, "test": 0}
    }

    for split_name in ["train", "validation", "test"]:
        real_split_folders = real_split[split_name]
        fake_split_folders = fake_split[split_name]

        if split_name == "train":
            selected = [("real", real_split_folders), ("fake", fake_split_folders)]
        else:
            min_count = min(len(real_split_folders), len(fake_split_folders))
            selected = [
                ("real", real_split_folders[:min_count]),
                ("fake", fake_split_folders[:min_count])
            ]

        for label, folders in selected:
            for subfolder_name, subfolder_path in folders:
                files = [f for f in os.listdir(subfolder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if not files:
                    continue
                random.shuffle(files)
                files = files[:max_images]

                for item in files:
                    dest_dir = {"train": train_dir, "validation": validation_dir, "test": test_dir}[split_name]
                    dest_path = os.path.join(dest_dir, label, subfolder_name, item)
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    source_path = os.path.join(subfolder_path, item)
                    shutil.copy2(source_path, dest_path)
                    counts[label][split_name] += 1

    print("\nBalanced Split Summary:")
    for label in counts:
        print(f"  {label.upper()} — Total: {sum(counts[label].values())}")
        for split in ["train", "validation", "test"]:
            print(f"    {split.capitalize()}: {counts[label][split]}")


if __name__ == "__main__":
    extract_all()

    source = r"D:\Celeb_DF\Frames"
    train_dir = r"D:\Celeb_DF\Frames\train"
    validation_dir = r"D:\Celeb_DF\Frames\validation"
    test_dir = r"D:\Celeb_DF\Frames\test"

    split_data(source, train_dir, validation_dir, test_dir)