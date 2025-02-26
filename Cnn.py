import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import cv2
import os
import shutil
import random

#------------------------------------------

def split_data(source, train_dir, validation_dir, test_dir, train_ratio=0.7, validation_ratio=0.15):
    for folder in [train_dir, validation_dir, test_dir]:
        os.makedirs(os.path.join(folder, "real"), exist_ok=True)
        os.makedirs(os.path.join(folder, "fake"), exist_ok=True)

    for category in ["real", "fake"]:
        category_path = os.path.join(source, category)
        subcategories = os.listdir(category_path)

        for subcategory in subcategories:
            subcategory_path = os.path.join(category_path, subcategory)
            videos = os.listdir(subcategory_path)

            random.shuffle(videos)

            train_split = int(len(videos) * train_ratio)
            validation_split = train_split + int(len(videos) * validation_ratio)

            for i, video in enumerate(videos):
                video_path = os.path.join(subcategory_path, video)
                if i < train_split:
                    dest = os.path.join(train_dir, category, subcategory, video)
                elif i < validation_split:
                    dest = os.path.join(validation_dir, category, subcategory, video)
                else:
                    dest = os.path.join(test_dir, category, subcategory, video)
                shutil.copytree(video_path, dest)

source = r"D:\Celeb_DF\Frames"
train_dir = r"D:\Celeb_DF\Frames\train"
validation_dir = r"D:\Celeb_DF\Frames\validation"
test_dir = r"D:\Celeb_DF\Frames\test"

split_data(source, train_dir, validation_dir, test_dir)