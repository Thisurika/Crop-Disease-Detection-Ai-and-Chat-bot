# backend/train_cnn.py
"""
Train CNN model for plant disease classification
Uses your 16-class dataset
"""

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import json
import os

# ────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────
DATA_DIR = "backend/dataset/color"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15               # Start with 15, increase if needed
MODEL_SAVE_PATH = "plant_disease_model.h5"

# Load class indices (from preprocess)
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)
NUM_CLASSES = len(class_indices)
print(f"Training for {NUM_CLASSES} classes")

# ────────────────────────────────────────────────
# Data generators with augmentation
# ────────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest',
    validation_split=0.2  # 80% train, 20% val
)

train_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# ────────────────────────────────────────────────
# Build model (MobileNetV2 + fine-tuning)
# ────────────────────────────────────────────────
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
base_model.trainable = False  # Freeze base (unfreeze later if needed)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()  # Print model layers

# Callbacks: stop early if no improvement, save best model
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True)
]

# ────────────────────────────────────────────────
# Train
# ────────────────────────────────────────────────
print("\nStarting training...")
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=callbacks
)

print("\nTraining complete!")
print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")

# Save final model (even if not best)
model.save(MODEL_SAVE_PATH)
print(f"Model saved to: {MODEL_SAVE_PATH}")