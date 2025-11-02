from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import base64
import io
import sqlite3
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load pretrained model
model = load_model('face_emotionModel.h5')

# Emotion labels used by the model
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Emotion-to-message mapping
emotion_messages = {
    'Angry': "You look upset. Take a deep breath, everything will be fine.",
    'Disgust': "You seem displeased. What's bothering you?",
    'Fear': "You appear scared. Stay calm, you’re safe.",
    'Happy': "You look happy! Keep smiling!",
    'Sad': "You are frowning. Why are you sad?",
    'Surprise': "You look surprised! Something unexpected?",
    'Neutral': "You seem calm and relaxed."
}

# Initialize database
def init_db():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT,
                        email TEXT,
                        matric_no TEXT,
                        image_path TEXT,
                        predicted_emotion TEXT,
                        message TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )''')
    conn.commit()
    conn.close()

init_db()

# Function to detect face and predict emotion
def detect_emotion(image_path):
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return "No face detected", "Please upload a clearer image."

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi = roi_gray.astype('float') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        prediction = model.predict(roi)[0]
        label = emotion_labels[np.argmax(prediction)]
        message = emotion_messages[label]
        return label, message

    return "Unknown", "Could not detect emotion."

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        matric_no = request.form['matric_no']
        image_file = request.files.get('image')

        # Handle image from webcam (base64)
        webcam_image_data = request.form.get('webcam_image')
        filename = f"{matric_no}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        if webcam_image_data:
            # Decode base64 image
            img_data = base64.b64decode(webcam_image_data.split(',')[1])
            image = Image.open(io.BytesIO(img_data))
            image.save(image_path)
        elif image_file:
            image_file.save(image_path)
        else:
            return render_template('index.html', message="Please upload or capture an image.")

        # Predict emotion
        predicted_emotion, friendly_message = detect_emotion(image_path)

        # Save to database
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (name, email, matric_no, image_path, predicted_emotion, message) VALUES (?, ?, ?, ?, ?, ?)",
                       (name, email, matric_no, image_path, predicted_emotion, friendly_message))
        conn.commit()
        conn.close()

        return render_template('index.html',
                               name=name,
                               emotion=predicted_emotion,
                               message=friendly_message)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
