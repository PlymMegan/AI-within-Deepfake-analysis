import os
import numpy as np
import cv2
import shutil
import random
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.utils import Sequence

#----------------------------------------------------

class VideoFrameGenerator(Sequence):
    def __init__(self, video_dir, batch_size=32, image_size=(224, 224), num_frames=10, shuffle=True):
        self.video_dir = video_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_frames = num_frames
        self.shuffle = shuffle
        self.video_paths = []
        self.labels = []
        
        self._load_video_paths_and_labels()
        self.on_epoch_end()
        
    def _load_video_paths_and_labels(self):
        for label, subfolder in enumerate(['real', 'fake']):  # Adjust based on your structure
            subfolder_path = os.path.join(self.video_dir, subfolder)
            if not os.path.exists(subfolder_path):
                continue
            for video_folder in os.listdir(subfolder_path):
                video_folder_path = os.path.join(subfolder_path, video_folder)
                if os.path.isdir(video_folder_path):
                    self.video_paths.append(video_folder_path)
                    self.labels.append(label)
    
    def __len__(self):
        return int(np.floor(len(self.video_paths) / self.batch_size))
    
    def __getitem__(self, index):
        batch_video_paths = self.video_paths[index * self.batch_size: (index + 1) * self.batch_size]
        batch_labels = self.labels[index * self.batch_size: (index + 1) * self.batch_size]
        
        frames = []
        for video_path in batch_video_paths:
            video_frames = self._load_video_frames(video_path)
            frames.append(video_frames)
        
        return np.array(frames), np.array(batch_labels)
    
    def _load_video_frames(self, video_folder_path):
        frames = []
        for frame_filename in sorted(os.listdir(video_folder_path)):
            if frame_filename.endswith(('.jpg', '.jpeg', '.png')):
                frame_path = os.path.join(video_folder_path, frame_filename)
                img = image.load_img(frame_path, target_size=self.image_size)
                img_array = image.img_to_array(img)
                frames.append(img_array)
                if len(frames) >= self.num_frames:
                    break
        frames = np.array(frames)
        if frames.shape[0] < self.num_frames:
            padding = np.zeros((self.num_frames - frames.shape[0], *self.image_size, 3))
            frames = np.vstack([frames, padding])
        return frames
    
    def on_epoch_end(self):
        if self.shuffle:
            temp = list(zip(self.video_paths, self.labels))
            np.random.shuffle(temp)
            self.video_paths, self.labels = zip(*temp)

def create_cnn_lstm_model(input_shape=(10, 224, 224, 3)):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape[1:])
    base_model.trainable = False

    model = models.Sequential()
    model.add(layers.TimeDistributed(base_model, input_shape=input_shape))
    model.add(layers.TimeDistributed(layers.Flatten()))
    model.add(layers.LSTM(256, return_sequences=False))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    video_folder = r"D:\Celeb_DF"
    output_root = r"D:\Celeb_DF\Frames"
    
    extract_frames(video_folder, output_root)

    source = r"D:\Celeb_DF\Frames"
    train_dir = r"D:\Celeb_DF\Frames\train"
    validation_dir = r"D:\Celeb_DF\Frames\validation"
    test_dir = r"D:\Celeb_DF\Frames\test"

    split_data(source, train_dir, validation_dir, test_dir)

    model = create_cnn_lstm_model()

    train_generator = VideoFrameGenerator(train_dir, batch_size=32)
    validation_generator = VideoFrameGenerator(validation_dir, batch_size=32)

    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=20,
        steps_per_epoch=len(train_generator),
        validation_steps=len(validation_generator)
    )
    
    test_loss, test_accuracy = model.evaluate(test_generator, steps=(test_generator.samples // test_generator.batch_size)+ 1)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

y_true = test_generator.classes

y_pred_probs = model.predict(test_generator, steps=len(test_generator))  
y_pred = (y_pred_probs > 0.5).astype(int)

print("Classification Report:")
print(classification_report(y_true, y_pred))

fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

model.save('cnn_model.keras')

def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.show()

plot_training_history(history)

model.summary()
