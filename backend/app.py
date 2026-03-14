# backend/app.py
import sys
import os
import traceback

# Add project root (CHATBOT/) to sys.path so 'chat' module is found
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from flask import Flask, render_template, request, jsonify, send_from_directory
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import random

# Import real chatbot
from chat.chatbot import get_chat_response

app = Flask(__name__,
            static_folder='../frontend/static',
            template_folder='../frontend/templates')

# ────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────
MODEL_PATH = "plant_disease_model.h5"
CLASS_JSON = "class_indices.json"
UPLOAD_FOLDER = "../frontend/static/uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ────────────────────────────────────────────────
# LOAD MODEL & CLASSES – WITH FULL DEBUG
# ────────────────────────────────────────────────
model = None
class_names = []

print("\n" + "═" * 80)
print("MODEL LOADING DEBUG – START")
print(f"Current working directory: {os.getcwd()}")
print(f"Model file path (relative): {MODEL_PATH}")
print(f"Model file absolute path: {os.path.abspath(MODEL_PATH)}")
print(f"File exists? {os.path.exists(MODEL_PATH)}")

if os.path.exists(MODEL_PATH):
    try:
        file_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        print(f"File size: {file_size_mb:.2f} MB")
        if file_size_mb < 5:
            print("WARNING: File is too small – probably corrupted or empty")
    except Exception as size_err:
        print(f"Cannot get file size: {size_err}")
else:
    print("!!! MODEL FILE NOT FOUND !!!")
    print("Please run 'python train_cnn.py' to create it in backend/ folder")

print(f"TensorFlow version: {tf.__version__}")

try:
    print("Trying to load model now...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("MODEL LOADED SUCCESSFULLY!")
except Exception as load_err:
    print("!!! MODEL LOAD FAILED !!!")
    print(f"Error type: {type(load_err).__name__}")
    print(f"Error message: {str(load_err)}")
    print("Full traceback:")
    traceback.print_exc()
    print("\nPOSSIBLE FIXES:")
    print("1. Retrain model: python train_cnn.py")
    print("2. Delete old .h5 file and retrain")
    print("3. Reinstall TensorFlow: pip install tensorflow==2.15.0")
    print("4. Try absolute path: MODEL_PATH = r'C:\\Users\\User\\Desktop\\CHATBOT\\backend\\plant_disease_model.h5'")
    model = None

# Load classes
try:
    with open(CLASS_JSON, "r") as f:
        class_indices = json.load(f)
    class_names = list(class_indices.keys())
    print(f"Loaded {len(class_names)} classes")
except Exception as e:
    print(f"Class indices load failed: {e}")

print("═" * 80 + "\n")

# ────────────────────────────────────────────────
# ROUTES
# ────────────────────────────────────────────────

@app.route('/', methods=['GET'])
def landing():
    return render_template('landing.html')

@app.route('/upload', methods=['GET'])
def upload_page():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({
            "error": "Model not loaded - check terminal logs for details",
            "details": "Model failed to load during server startup. See terminal for error message."
        }), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = f"{random.randint(10000,99999)}_{file.filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        img = Image.open(filepath).convert("RGB").resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        pred = model.predict(img_array)[0]
        top_idx = np.argmax(pred)
        confidence = float(pred[top_idx]) * 100
        disease_raw = class_names[top_idx]
        disease = disease_raw.replace("_", " ")

        advice = get_chat_response(
            "Give practical treatment and prevention advice for this disease in Sri Lanka",
            disease=disease_raw
        )

        return jsonify({
            "disease": disease,
            "confidence": round(confidence, 2),
            "image_url": f"uploads/{filename}",
            "advice": advice
        })

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    query = data.get('query')
    disease = data.get('disease')

    if not query:
        return jsonify({"error": "No question provided"}), 400

    try:
        response = get_chat_response(query, disease=disease)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"response": f"Error connecting to AI: {str(e)}. Try again or consult local expert."})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')