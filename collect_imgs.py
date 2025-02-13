import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Updated labels for 13 classes
labels = ["L", "B", "A", "Hello", "Yes", "No", "Stop", "Please", "Right", "Left", "Thumbs Up", "Thumbs Down"]
number_of_classes = len(labels)
dataset_size = 200  # Increase dataset size for better accuracy

cap = cv2.VideoCapture(0)

for j, label in enumerate(labels):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class: {label} ({j}/{number_of_classes})')

    # Wait for user confirmation
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, f'Collecting: {label} - Press "Q" to start', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)
        counter += 1
        cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
