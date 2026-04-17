# Crop-Disease-Detection-Ai-and-Chat-bot
# 🌱 Plant AI - Plant Disease Detection & Advisor

A smart web application that helps farmers detect plant diseases from leaf images and get practical treatment advice using AI.

## 🌱 Plant AI Demo

Here are some screenshots of the application:

![Screenshot 1](sc/1.png)
![Screenshot 2](sc/2.png)
![Screenshot 3](sc/3.png)
![Screenshot 4](sc/4.png)
![Screenshot 5](sc/5.png)

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

- 
---

## ⚙️ Installation & Setup

### 1. Clone repository

### 2. Create virtual environment

### 3. Install dependencies

### 4. Add dataset
- Download dataset separately
- Place it inside `backend/dataset/`

---

## ▶️ Run Project

Then open: https://127.0.0:5000


---

## ⚠️ Notes
- Dataset is not included due to size limitations
- Virtual environment is ignored in GitHub
- Ensure TensorFlow version matches requirements

---

## 👨‍💻 Author
Thisurika Hasajith
SLIIT Student  

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
├── .env                          # Api Keys
├──  venv folder                 
└── README.md


