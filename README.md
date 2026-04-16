# Crop-Disease-Detection-Ai-and-Chat-bot
# 🌱 Plant AI - Plant Disease Detection & Advisor

A smart web application that helps farmers detect plant diseases from leaf images and get practical treatment advice using AI.

![Plant AI Demo](https://via.placeholder.com/800x400?text=Plant+AI+Demo)  
*(Replace with your actual screenshot)*

## ✨ Features

- **Leaf Image Upload** with drag & drop support.
- **Disease Detection** using Deep Learning (MobileNetV2)
- **Confidence Score** with visual progress bar
- **Expert Advice** powered by Google Gemini AI + FAQ system
- **Interactive Chat** - Ask any questions about treatment or prevention
- **Beautiful & Responsive UI** with green agricultural theme
- **Practical Sri Lanka-focused advice** (neem oil, cow urine, monsoon tips, etc.)

## 🛠️ Tech Stack

### Backend
- **Python** + **Flask**
- **TensorFlow / Keras** (MobileNetV2 - Transfer Learning)
- **Google Gemini API** (for intelligent advice)
- **Pandas** (FAQ system)

### Frontend
- **HTML5, CSS3, JavaScript**
- Drag & Drop Upload
- Real-time Chat Interface

### Dataset
- PlantVillage Dataset (15 classes: Tomato, Potato, Pepper diseases)

## 📂 Project Structure

```bash
CHATBOT/
├── backend/
│   ├── app.py                    # Main Flask app
│   ├── train_cnn.py              # Model training script
│   ├── preprocess_explore.py     # Data preprocessing & EDA
│   ├── plant_disease_model.h5    # Trained model
│   ├── class_indices.json        # Class mapping
│   └── chat/
│       └── chatbot.py            # Gemini + FAQ logic
├── frontend/
│   ├── templates/
│   │   ├── index.html
│   │   └── landing.html
│   └── static/
│       ├── css/style.css
│       └── js/script.js
├── dataset/color/                # Your 15 class folders
├── .env                          # API keys
└── README.md
