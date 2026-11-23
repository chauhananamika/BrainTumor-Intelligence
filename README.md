# 🧠 NeuroScan — Brain MRI Tumor Classifier

A web application that classifies 2D brain MRI slices into **Glioma**, **Meningioma**, **Pituitary tumor**, or **No tumor** using a Convolutional Neural Network (CNN), with Grad-CAM heatmap visualization.

<p align="center">
  <img width="404" alt="NeuroScan Demo" src="https://github.com/user-attachments/assets/25e9b470-f046-46cd-a629-ea67423c419c" />
</p>

---

## ✨ Features

| Feature | Description |
|---|---|
| **4-Class Classification** | Glioma · Meningioma · Pituitary · No tumor |
| **Grad-CAM Heatmap** | Jet-colormap attention overlay highlighting the regions driving the prediction |
| **Heatmap Controls** | Toggle On/Off + real-time opacity slider |
| **Drag & Drop Upload** | Drop an MRI image, click to browse, or Ctrl+V paste from clipboard |
| **Sample Test Set** | 5 built-in MRI scans to try instantly (Glioma, Meningioma, Pituitary, No tumor, Hard case) |
| **Calibrated Probabilities** | Animated probability bars for all 4 classes |
| **Stats Dashboard** | Confidence %, distribution check, latency, model version |
| **Dark Theme UI** | Professional medical-grade dark interface |

---

## 🗂️ Project Structure

```
Brain-Tumor-Detection-CNN-Architecture/
├── app.py                      # Flask backend + CNN model + GradCAM
├── Brain_tumor_predict.ipynb   # Jupyter notebook (training code)
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
├── templates/
│   ├── base.html               # Base layout (navbar, scripts)
│   └── index.html              # Main classifier page
├── static/
│   ├── css/main.css            # Dark theme styling
│   ├── js/main.js              # Upload, AJAX, heatmap controls
│   └── samples/                # Preset test MRI images
│       ├── glioma.jpg
│       ├── meningioma.jpg
│       ├── pituitary.jpg
│       ├── no_tumor.jpg
│       └── hard_case.jpg
├── uploads/                    # User uploaded images (runtime)
└── img/                        # README screenshots
```

---

## 🚀 Setup & Run

### 1. Clone the Repository

```bash
git clone https://github.com/chauhananamika/BrainTumor-Intelligence.git
cd BrainTumor-Intelligence
```

### 2. Create Virtual Environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Web App

```bash
python app.py
```

Open **[https://braintumor-intelligence.onrender.com/]** in your browser.

> **Note:** On first run, the app auto-generates a baseline `brain_tumor.h5` model. For a trained model with real accuracy, run the `Brain_tumor_predict.ipynb` notebook with the Kaggle dataset first.

---

## 📊 Model Architecture

- **Base**: CNN with Conv2D → MaxPooling → Dense layers
- **Input Size**: 224 × 224 × 3 (RGB)
- **Output**: 4-class softmax (Glioma, Meningioma, No tumor, Pituitary)
- **Training Notebook**: Uses MobileNetV2 transfer learning via TensorFlow Hub

---

## 📁 Dataset

[Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection) — Kaggle (Masoud Nickparvar, CC BY 4.0)

---

## 🛠️ Tech Stack

- **Backend**: Python, Flask
- **Deep Learning**: TensorFlow / Keras
- **Frontend**: HTML5, CSS3, JavaScript, jQuery
- **Visualization**: Grad-CAM heatmap (Jet colormap)

---

## ⚠️ Disclaimer

> **Research demo, not a medical device.** Results must not be used for diagnosis or treatment decisions.

---

## 📝 License

This project is for educational and research purposes only.
