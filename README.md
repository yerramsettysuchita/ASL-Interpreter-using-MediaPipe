# ASL-Interpreter-using-MediaPipe
📦 Sign-Language-Recognition

### 🛠 Tech Stack
- Python
- Flask
- OpenCV
- Mediapipe
- Scikit-learn
- Pickle
- NumPy

### 📁 Repository Structure
```
📦 Sign-Language-Recognition
 ┣ 📂 data
 ┃ ┗ 📄 data.pickle
 ┣ 📂 models
 ┃ ┗ 📄 model.p
 ┣ 📂 static
 ┣ 📂 templates
 ┃ ┣ 📄 index.html
 ┃ ┣ 📄 camera.html
 ┣ 📄 app.py
 ┣ 📄 train_classifier.py
 ┣ 📄 inference_classifier.py
 ┣ 📄 create_dataset.py
 ┣ 📄 collect_imgs.py
 ┣ 📄 requirements.txt
 ┣ 📄 README.md
```

## 📖 Project Overview
Sign Language Recognition is an AI-powered system designed to interpret hand gestures and convert them into meaningful text. The project leverages computer vision and machine learning techniques to help bridge the communication gap between individuals using sign language and those unfamiliar with it. Using the power of OpenCV and Mediapipe, it captures hand gestures and classifies them using a trained machine learning model built with Scikit-learn. The system supports multiple sign classes and provides real-time recognition via a web-based application using Flask.

## ✨ Features
- Real-time sign language detection.
- Hand landmark tracking with Mediapipe.
- Machine learning classification using RandomForest.
- Flask-based web application for user interaction.
- Supports various common sign language gestures.
- Easy dataset collection and training pipeline.

## 📜 Files & Descriptions

#### **`app.py`**
- Runs a Flask server for the web application.
- Captures real-time video and predicts sign language gestures.
- Uses `mediapipe` for hand tracking.
- Loads a trained model (`model.p`) for inference.

#### **`train_classifier.py`**
- Loads dataset (`data.pickle`) and trains a `RandomForestClassifier`.
- Performs data splitting (train, validation, test).
- Evaluates the model using cross-validation and saves the trained model.

#### **`inference_classifier.py`**
- Loads the trained model (`model.p`).
- Captures real-time video, processes hand landmarks, and predicts signs.
- Uses OpenCV and Mediapipe for visualization.

#### **`create_dataset.py`**
- Reads images from `data/` directory.
- Extracts hand landmarks and stores them in `data.pickle`.

#### **`collect_imgs.py`**
- Captures images using a webcam for different sign classes.
- Saves them in `data/` directory.

#### **`requirements.txt`**
To install dependencies:
```bash
pip install -r requirements.txt
```

## 📜 Project Description (200 Words)
Sign Language Recognition is an AI-powered application aimed at bridging the communication gap for individuals who use sign language. By leveraging deep learning and computer vision, this system captures real-time hand gestures through a webcam, processes them using Mediapipe, and classifies them with a machine learning model built using Scikit-learn. The application is designed to identify a range of commonly used sign language gestures and translate them into readable text, enabling seamless interaction between users.

The project consists of a complete pipeline from data collection to model training and real-time inference. The `collect_imgs.py` script enables users to capture and label training data, while `create_dataset.py` extracts hand landmark features for model training. The classifier, trained using RandomForest, achieves high accuracy and is deployed using Flask for an interactive web-based experience.

By integrating AI-driven solutions with assistive technology, this project enhances accessibility for individuals with hearing or speech impairments. It has applications in educational tools, accessibility solutions, and human-computer interaction. The framework is modular, allowing easy expansion with additional gestures or sign languages in future iterations. This open-source project invites collaboration to further improve communication technology for the deaf community.

## 🚀 Deployment Guide
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Train the model:
   ```bash
   python train_classifier.py
   ```
3. Run the Flask app:
   ```bash
   python app.py
   ```
4. Open `http://127.0.0.1:5000/` in a browser.
