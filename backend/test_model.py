# backend/test_model.py
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

MODEL_PATH = "plant_disease_model.h5"
CLASS_JSON = "class_indices.json"

# Load model and classes
model = tf.keras.models.load_model(MODEL_PATH)
with open(CLASS_JSON, "r") as f:
    class_indices = json.load(f)
class_names = list(class_indices.keys())

# Test image 
test_image_path = "\dataset\color\Pepper__bell___Bacterial_spot"  

img = Image.open(test_image_path).resize((224, 224))
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

pred = model.predict(img_array)[0]
top_idx = np.argmax(pred)
confidence = pred[top_idx] * 100

print(f"Predicted: {class_names[top_idx]}")
print(f"Confidence: {confidence:.2f}%")

# Top 3
top3 = np.argsort(pred)[-3:][::-1]
print("\nTop 3:")
for i in top3:
    print(f"  {class_names[i]:<40} {pred[i]*100:.2f}%")