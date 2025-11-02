from flask import Flask, render_template, request
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import base64
import io
import sqlite3
from datetime import datetime

# ----------------------------
# Flask App Initialization
# ----------------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ----------------------------
# Load TensorFlow Lite Model
# ----------------------------
model_path = "face_emotionModel_compat.tflite"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' not found. Please ensure it’s in the same folder as app.py")

print("✅ Loading TensorFlow Lite model...")
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details for inference
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("✅ Model loaded successfully!")

# ----------------------------
# Emotion Labels & Messages
# ----------------------------
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
emotion_messages = {
    'Angry': "You look upset. Take a deep breath, everything will be fine.",
    'Disgust': "You seem displeased. What's bothering you?",
    'Fear': "You appear scared. Stay calm, you’re safe.",
    'Happy': "You look happy! Keep smiling!",
    'Sad': "You are frowning. Why are you sad?",
    'Surprise': "You look surprised! Something unexpected?",
    'Neutral': "You seem calm and relaxed."
}

# ----------------------------
# Initialize Database
# ----------------------------
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

# ----------------------------
# Emotion Detection Function
# ----------------------------
def detect_emotion(image_path):
    # Load Haar Cascade for face detection
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return "No face detected", "Please upload a clearer image."

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi = roi_gray.astype('float32') / 255.0
        roi = np.expand_dims(roi, axis=-1)
        roi = np.expand_dims(roi, axis=0)

        # Convert grayscale to RGB if model expects 3 channels
        if input_details[0]['shape'][-1] == 3:
            roi = np.repeat(roi, 3, axis=-1)

        # Perform prediction
        interpreter.set_tensor(input_details[0]['index'], roi)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0]

        label = emotion_labels[np.argmax(prediction)]
        message = emotion_messages[label]
        return label, message

    return "Unknown", "Could not detect emotion."

# ----------------------------
# Flask Routes
# ----------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        matric_no = request.form['matric_no']
        image_file = request.files.get('image')

        # Handle webcam image
        webcam_image_data = request.form.get('webcam_image')
        filename = f"{matric_no}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        if webcam_image_data:
            img_data = base64.b64decode(webcam_image_data.split(',')[1])
            image = Image.open(io.BytesIO(img_data))
            image.save(image_path)
        elif image_file:
            image_file.save(image_path)
        else:
            return render_template('index.html', message="Please upload or capture an image.")

        # Predict emotion
        predicted_emotion, friendly_message = detect_emotion(image_path)

        # Save result in DB
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (name, email, matric_no, image_path, predicted_emotion, message) VALUES (?, ?, ?, ?, ?, ?)",
                       (name, email, matric_no, image_path, predicted_emotion, friendly_message))
        conn.commit()
        conn.close()

        return render_template('index.html', name=name, emotion=predicted_emotion, message=friendly_message)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
