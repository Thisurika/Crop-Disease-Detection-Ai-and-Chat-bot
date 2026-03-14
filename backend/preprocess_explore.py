# backend/preprocess_explore.py
"""
Plant Disease Dataset Exploration & Preprocessing Setup
Works with 15–16 class subset (automatically detects folders)
"""

import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array, array_to_img

# ────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────
DATA_DIR = "backend/dataset/color"
IMG_SIZE = (224, 224)
SAVE_CLASS_INDICES = True
CLASS_INDICES_PATH = "class_indices.json"

# Optional: skip folders that are not real classes (e.g., 'PlantVillage')
SKIP_FOLDERS = {'PlantVillage'}  # add any other wrapper folders here

# ────────────────────────────────────────────────
# Debug: Check folder & classes
# ────────────────────────────────────────────────
print("Current working directory:", os.getcwd())
print("DATA_DIR path:", os.path.abspath(DATA_DIR))
print("DATA_DIR exists?", os.path.exists(DATA_DIR))

if not os.path.exists(DATA_DIR):
    print("ERROR: Folder not found. Please check path or move dataset.")
    exit(1)

all_dirs = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
classes = [d for d in all_dirs if d not in SKIP_FOLDERS]

print(f"Found {len(all_dirs)} total folders, using {len(classes)} classes after filtering.")
print("Classes:", classes)

if len(classes) == 0:
    print("ERROR: No valid class folders found.")
    exit(1)

# ────────────────────────────────────────────────
# 1. Show 5 original images
# ────────────────────────────────────────────────
def show_original_images():
    print("\n=== Showing 5 original images (before preprocessing) ===")

    selected_classes = random.sample(classes, min(5, len(classes)))

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    fig.suptitle("5 Original Images – Before Preprocessing", fontsize=16)

    for ax, cls in zip(axes.flat, selected_classes):
        class_path = os.path.join(DATA_DIR, cls)
        img_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not img_files:
            ax.text(0.5, 0.5, "No images", ha='center')
            continue

        img_path = os.path.join(class_path, random.choice(img_files))
        try:
            img = Image.open(img_path)
            ax.imshow(img)
            ax.set_title(f"{cls}\n{img.size}", fontsize=10)
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {e}", ha='center')
        ax.axis('off')

    plt.tight_layout()
    plt.show()


# ────────────────────────────────────────────────
# 2. Augmentation generator
# ────────────────────────────────────────────────
def get_augmentation_datagen():
    return ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )


# ────────────────────────────────────────────────
# 3. Show original + augmented
# ────────────────────────────────────────────────
def show_augmented_examples(num_examples=5, num_augs_per_img=4):
    print(f"\n=== Showing {num_examples} originals + {num_augs_per_img} augmented each ===")

    datagen = get_augmentation_datagen()

    selected_classes = random.sample(classes, min(num_examples, len(classes)))

    fig, axes = plt.subplots(num_examples, num_augs_per_img + 1, figsize=(18, 3 * num_examples))
    fig.suptitle("Original vs Augmented Versions", fontsize=16)

    for row, cls in enumerate(selected_classes):
        class_path = os.path.join(DATA_DIR, cls)
        img_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not img_files:
            continue

        img_path = os.path.join(class_path, random.choice(img_files))

        # Original
        img = load_img(img_path, target_size=IMG_SIZE)
        axes[row, 0].imshow(img)
        axes[row, 0].set_title(f"Original\n{cls}", fontsize=9)
        axes[row, 0].axis('off')

        # Augmented
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        i = 0
        for batch in datagen.flow(img_array, batch_size=1):
            aug_img = array_to_img(batch[0])
            axes[row, i + 1].imshow(aug_img)
            axes[row, i + 1].set_title(f"Aug {i+1}", fontsize=9)
            axes[row, i + 1].axis('off')
            i += 1
            if i >= num_augs_per_img:
                break

    plt.tight_layout()
    plt.show()


# ────────────────────────────────────────────────
# 4. EDA
# ────────────────────────────────────────────────
def perform_eda():
    print("\n=== EDA: Class Distribution & Image Stats ===")

    class_counts = []
    avg_areas = []

    for cls in classes:
        class_path = os.path.join(DATA_DIR, cls)
        files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        count = len(files)
        class_counts.append(count)

        areas = []
        for f in random.sample(files, min(20, len(files))):
            try:
                with Image.open(os.path.join(class_path, f)) as img:
                    areas.append(img.width * img.height / 1000)
            except:
                pass
        avg_areas.append(np.mean(areas) if areas else 0)

    df = pd.DataFrame({
        'Class': classes,
        'Image Count': class_counts,
        'Avg Area (k pixels)': avg_areas
    })

    plt.figure(figsize=(10, 6))
    sns.boxplot(y=df['Image Count'], color='lightgreen')
    plt.title(f"Boxplot – Images per Class ({len(classes)} classes)")
    plt.ylabel("Number of Images")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    plt.figure(figsize=(12, 6))
    top = df.sort_values('Image Count', ascending=False).head(10)
    sns.barplot(data=top, x='Image Count', y='Class', palette='viridis')
    plt.title("Top 10 Classes by Image Count")
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.histplot(df['Avg Area (k pixels)'], bins=15, kde=True, color='teal')
    plt.title("Average Image Area Distribution")
    plt.xlabel("Area (thousands of pixels)")
    plt.show()

    print("\nSummary Statistics:")
    print(df.describe().round(1))
    print("\nAll classes (sorted by count):")
    print(df.sort_values('Image Count', ascending=False))


# ────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Plant Disease Preprocessing & Exploration ===\n")

    show_original_images()
    show_augmented_examples(num_examples=5, num_augs_per_img=4)
    perform_eda()

    if SAVE_CLASS_INDICES:
        class_indices = {cls: i for i, cls in enumerate(classes)}
        with open(CLASS_INDICES_PATH, "w") as f:
            json.dump(class_indices, f, indent=2)
        print(f"\nSaved {len(classes)} class indices → {CLASS_INDICES_PATH}")

    print("\nExploration finished. Close plot windows when ready.")
    plt.show(block=True)