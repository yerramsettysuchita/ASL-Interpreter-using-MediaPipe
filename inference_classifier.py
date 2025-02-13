import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
from PIL import Image, ImageDraw, ImageFont

# Load trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Labels dictionary
labels_dict = {
    0: 'L', 1: 'B', 2: 'A', 3: 'Hello', 4: 'Yes', 5: 'No',
    6: 'Stop', 7: 'Please', 8: 'Right', 9: 'Left',
    10: 'Thumbs Up', 11: 'Thumbs Down'
}

# Function to render text with Unicode support
def put_text_unicode(img, text, position, font_size=48, color=(0, 255, 0)):
    """ Draws text on OpenCV images using PIL (Unicode Support) """
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)

    # Load font for English text
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()  # Use default if the font is unavailable

    draw.text(position, text, font=font, fill=color)
    return np.array(img_pil)

# Start Timer
start_time = time.time()
cap = cv2.VideoCapture(0)

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]  # Take only ONE hand (first detected)
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Collect features from the first detected hand only
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

        # Ensure the feature count is correct
        if len(data_aux) == 42:
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) + 10
            y2 = int(max(y_) * H) + 10

            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            # Use PIL to render English text (larger font size)
            text_to_display = f'{predicted_character}'

            # Draw prediction box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)

            # Use PIL to render text correctly
            frame = put_text_unicode(frame, text_to_display, (x1, y1 - 50), font_size=60, color=(0, 255, 0))

    # Show frame
    cv2.imshow('Sign Language Recognition', frame)

    # Check if 1 minute (60 seconds) has passed
    if time.time() - start_time > 60:
        print("Time limit reached! Closing...")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
