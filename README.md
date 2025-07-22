# 🏏🏆 Sports Celebrity Image Classifier

A machine learning project that classifies images of popular sports celebrities using classical ML techniques (SVM, Random Forest, Logistic Regression) with facial feature extraction via OpenCV and wavelet transforms. The project is deployed using Flask for easy API access.

---

## 🚀 Project Overview

This project builds an image classifier for 5 sports celebrities:
- **Lionel Messi**
- **Maria Sharapova**
- **Roger Federer**
- **Serena Williams**
- **Virat Kohli**

The pipeline:
1. Detects faces and eyes using Haar cascades.
2. Crops the facial region if at least two eyes are detected.
3. Applies wavelet transform for feature extraction.
4. Trains and tunes multiple ML models (SVM, RandomForest, Logistic Regression).
5. Deploys the best model via a Flask API.

---

## 🛠️ Tech Stack

- **Python 3**
- **OpenCV**
- **scikit-learn**
- **Wavelet Transform (pywt)**
- **Flask**
- **Joblib**
- **NumPy, Pandas, Matplotlib, Seaborn**

---

## 🧠 Workflow

### 🔍 1. Preprocessing
- Load images from `./dataset/`.
- Use Haar cascades (`haarcascade_frontalface_default.xml` and `haarcascade_eye.xml`) to detect faces.
- Crop the image if 2+ eyes are found.
- Save the cropped face images into `./dataset/cropped/`.

### 🧾 2. Feature Engineering
Each image is converted into:
- A 32x32 color image.
- A wavelet-transformed grayscale image.
These are combined into a single feature vector of length **4096**.

### 🤖 3. Model Training
Train on multiple models:
- **SVM (Support Vector Classifier)**
- **Random Forest**
- **Logistic Regression**

Model selection is done via **GridSearchCV** using 5-fold cross-validation.

### 📦 4. Artifact Saving
- Best model is saved as `saved_model.pkl`.
- Class mapping is saved as `class_dictionary.json`.

---

## 🔥 Flask API

### Endpoint:
```bash
POST /classify_image
```
---

## 📂 Project Structure
project/

│

├── dataset/                                        # Raw training images per celebrity

│   └── cropped/                                    # Auto-generated cropped face images

│
├── haarcascades/                                   # Haar cascade XMLs for face/eye detection

│

├── test_images/                                    # For testing face/eye detection

├── test_image/                                     # Test images used in util.py

│

├── artifacts/

│   ├── saved_model.pkl                             # Trained best model (SVM)

│   └── class_dictionary.json                       # Class-label mapping

│

├── server/

│   ├── server.py                                   # Flask server

│   ├── util.py                                     # Classification logic + image preprocessing

│   └── b64.txt                                     # Base64 encoded test image
│
├── Model.py                        # Main ML training pipeline
└── wavelet.py                      # Wavelet transformation logic


