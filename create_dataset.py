import os
import pickle
import mediapipe as mp
import cv2
import numpy as np

# Initialize Mediapipe Hand Detector
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

# Updated labels corresponding to new sign classes
labels = ["L", "B", "A", "Hello", "Yes", "No", "Stop", "Please", "Right", "Left", "Thumbs Up", "Thumbs Down"]
data = []
labels_data = []

for label_idx, label in enumerate(labels):
    class_dir = os.path.join(DATA_DIR, str(label_idx))
    for img_path in os.listdir(class_dir):
        img = cv2.imread(os.path.join(class_dir, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                data_aux = []
                x_ = []
                y_ = []

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                data.append(data_aux)
                labels_data.append(label_idx)

# Save the dataset to a file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels_data}, f)

print("Dataset creation completed successfully! ✅")
