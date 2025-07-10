#!/usr/bin/env python
# coding: utf-8

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[3]:


import os
import subprocess
import cv2
from mtcnn import MTCNN
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Sequential

def extract_key_frames(video_path, temp_dir):
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-vf', "select='eq(pict_type\\,I)'",
        '-vsync', 'vfr',
        os.path.join(temp_dir, 'frame_%03d.png')
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def detect_and_crop_face(image_path, output_path):
    detector = MTCNN()
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(image_rgb)
    if len(faces) > 0:
        faces = sorted(faces, key=lambda x: x['box'][2] * x['box'][3], reverse=True)
        box = faces[0]['box']
        x, y, w, h = box
        padding = 20
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)
        cropped = image[y1:y2, x1:x2]
        cropped_resized = cv2.resize(cropped, (299, 299))
        cv2.imwrite(output_path, cropped_resized)
    else:
        print(f"No face detected in {image_path}")

train_videos = list(range(1, 41))
val_videos = list(range(41, 54))

for split, videos in [('train', train_videos), ('val', val_videos)]:
    for label in ['real', 'fake']:
        output_dir = f'processed/{split}/{label}'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for video_num in videos:
            if label == 'real':
                video_path = f'/content/drive/My Drive/SDFVD/videos_real/v{video_num}.mp4'
            else:
                video_path = f'/content/drive/My Drive/SDFVD/videos_fake/vs{video_num}.mp4'
            temp_dir = 'temp_keyframes'
            extract_key_frames(video_path, temp_dir)
            for frame_file in os.listdir(temp_dir):
                frame_path = os.path.join(temp_dir, frame_file)
                output_frame_path = os.path.join(output_dir, f'video{video_num}_{frame_file}')
                detect_and_crop_face(frame_path, output_frame_path)
            for file in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, file))
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=20,
    zoom_range=0.2
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'processed/train',
    target_size=(299, 299),
    batch_size=32,
    class_mode='categorical'
)

val_real_frames = [f for f in os.listdir('processed/val/real') if os.path.isfile(os.path.join('processed/val/real', f))]
val_fake_frames = [f for f in os.listdir('processed/val/fake') if os.path.isfile(os.path.join('processed/val/fake', f))]
val_df = pd.DataFrame({
    'filename': [os.path.join('processed/val/real', f) for f in val_real_frames] +
                [os.path.join('processed/val/fake', f) for f in val_fake_frames],
    'class': ['real'] * len(val_real_frames) + ['fake'] * len(val_fake_frames),
    'video_id': [f'video{f.split("_")[0].replace("video", "")}_{"real" if "real" in f else "fake"}' for f in val_real_frames + val_fake_frames]
})
val_generator = val_datagen.flow_from_dataframe(
    val_df,
    x_col='filename',
    y_col='class',
    target_size=(299, 299),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

base_model = Xception(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.5),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])
base_model.trainable = False
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_generator, epochs=10, validation_data=val_generator)

val_predictions = model.predict(val_generator)
val_pred_labels = np.argmax(val_predictions, axis=1)
val_true_labels = val_generator.classes

frame_accuracy = accuracy_score(val_true_labels, val_pred_labels)
print(f'Frame-level accuracy: {frame_accuracy:.4f}')

video_predictions = {}
for i, video_id in enumerate(val_df['video_id']):
    if video_id not in video_predictions:
        video_predictions[video_id] = []
    video_predictions[video_id].append(val_pred_labels[i])

video_pred_labels = {}
for video_id, preds in video_predictions.items():
    video_pred_labels[video_id] = 1 if 1 in preds else 0

video_true_labels = {vid: 0 if vid.endswith('_real') else 1 for vid in video_predictions.keys()}
correct = sum(1 for vid, pred in video_pred_labels.items() if pred == video_true_labels[vid])
video_accuracy = correct / len(video_pred_labels)
print(f'Video-level accuracy: {video_accuracy:.4f}')

