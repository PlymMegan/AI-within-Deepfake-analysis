import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import cv2
import os
import shutil
import random
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
import numpy as np
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import Sequence
from sklearn.utils import shuffle
from tensorflow.keras import regularizers
import gc
from tensorflow.keras import backend as K
import math

#------------------------------------------

class SequenceDataGenerator(Sequence):
    def __init__(self, directory, batch_size=4, sequence_length=10, target_size=(224, 224), shuffle_data=True):
        self.directory = directory
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.target_size = target_size
        self.shuffle_data = shuffle_data
        self.class_map = {"real": 1, "fake": 0}
        self.samples = self._load_samples()
        if self.shuffle_data:
            self.samples = shuffle(self.samples)

    def _load_samples(self):
        samples = []
        for class_label in ["real", "fake"]:
            class_dir = os.path.join(self.directory, class_label)
            for folder_name in os.listdir(class_dir):
                folder_path = os.path.join(class_dir, folder_name)
                if os.path.isdir(folder_path):
                    samples.append((folder_path, self.class_map[class_label]))
        return samples

    def __len__(self):
        return len(self.samples) // self.batch_size

    def __getitem__(self, idx):
        batch_samples = self.samples[idx * self.batch_size:(idx + 1) * self.batch_size]
        X_batch = np.zeros((self.batch_size, self.sequence_length, *self.target_size, 3), dtype=np.float32)
        y_batch = np.zeros((self.batch_size,), dtype=np.int32)

        for i, (folder_path, label) in enumerate(batch_samples):
            frame_files = sorted([
                f for f in os.listdir(folder_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ], key=lambda x: int(x.split('_')[-1].split('.')[0]))  # sort by frame index

            frame_files = frame_files[:self.sequence_length]
            for j, file_name in enumerate(frame_files):
                img = cv2.imread(os.path.join(folder_path, file_name))
                img = cv2.resize(img, self.target_size)
                img = img.astype(np.float32) / 255.0
                X_batch[i, j] = img

            if len(frame_files) < self.sequence_length:
                pad_amt = self.sequence_length - len(frame_files)
                X_batch[i, len(frame_files):] = 0.0

            y_batch[i] = label

        return X_batch, y_batch

    def on_epoch_end(self):
        if self.shuffle_data:
            self.samples = shuffle(self.samples)

def create_cnn_lstm(sequence_length=10):
    cnn_base = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.GlobalAveragePooling2D()
    ])

    model = models.Sequential([
        layers.TimeDistributed(cnn_base, input_shape=(sequence_length, 224, 224, 3)),
        layers.Bidirectional(layers.LSTM(64, return_sequences=False)),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


train_dir = r"D:\Celeb_DF\Frames\train"
val_dir = r"D:\Celeb_DF\Frames\validation"
test_dir = r"D:\Celeb_DF\Frames\test"

train_generator = SequenceDataGenerator(train_dir, batch_size=8)
val_generator = SequenceDataGenerator(val_dir, batch_size=8)
test_generator = SequenceDataGenerator(test_dir, batch_size=8, shuffle_data=False)

model = create_cnn_lstm(sequence_length=10)
model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, verbose=1)

train_labels = [label for _, label in train_generator.samples]

class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weights = dict(enumerate(class_weights_array))
print("Class weights:", class_weights, flush=True)

val_labels = [label for _, label in val_generator.samples]
num_real = sum(val_labels)
num_fake = len(val_labels) - num_real
print(f"Validation set: {num_real} real, {num_fake} fake", flush=True)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=[early_stopping, lr_scheduler, model_checkpoint],
    class_weight=class_weights
)

print("\nPredicting on test set...")

steps = math.ceil(len(test_generator.samples) / test_generator.batch_size)
y_pred_probs = model.predict(test_generator, steps=steps, verbose=1)

if len(y_pred_probs) != len(test_generator.samples):
    print(f"Warning: Predicted {len(y_pred_probs)} samples but true labels are {len(test_generator.samples)}. Fixing...")
    y_pred_probs = y_pred_probs[:len(test_generator.samples)]

y_pred = (y_pred_probs > 0.5).astype(int).flatten()
y_true = np.array([label for _, label in test_generator.samples])

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['Fake', 'Real']))

fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.grid()
plt.show()

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


model.save("cnn_model.keras")

del model
K.clear_session()
gc.collect()

print("History keys:", history.history.keys())

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
