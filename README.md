# 🧠 NeuroScan — Brain MRI Tumor Classifier

NeuroScan is an AI-powered brain MRI tumor classification web application built with **TensorFlow / Keras** and **Flask**. It analyzes axial 2D brain MRI slices and classifies them into four categories: **Glioma**, **Meningioma**, **Pituitary Tumor**, and **No Tumor**, accompanied by real-time **Grad-CAM (Gradient-weighted Class Activation Mapping)** visualizations to explain model predictions.

---

## ✨ Features

- **Four-Class Brain MRI Classification:** Detects Glioma, Meningioma, Pituitary tumors, and healthy non-tumor scans.
- **Genuine MobileNetV2 Transfer Learning:** Uses ImageNet-pretrained feature extractors with a dedicated classification head.
- **Model Confidence & Class Probabilities:** Displays exact winning softmax confidence and animated probability distribution bars for all four classes.
- **Explainable AI (Grad-CAM):** Generates gradient-weighted class activation heatmaps backpropagated from the final convolutional layer (`out_relu`) to show which anatomical regions drove the classification.
- **Interactive Heatmap Controls:** Real-time visibility toggle (on/off) and opacity slider ($0\%$ to $100\%$) overlaid directly onto the MRI scan.
- **Built-in Test Scans:** Five pre-loaded sample MRI scans (Glioma, Meningioma, Pituitary, No tumor, and complex cases) for instant testing.
- **Versatile Input Methods:** Drag-and-drop zone, file browser, and clipboard paste (`Ctrl+V`) supporting PNG, JPEG, and BMP formats up to 10 MB.
- **Responsive Dark Theme UI:** Streamlined clinical-inspired dashboard.

---

## 🔬 Model Architecture & Processing Pipeline

1. **Input Dimensions:** Images are loaded, converted to RGB, and resized to $224 \times 224 \times 3$.
2. **Integrated Normalization:** An internal `layers.Rescaling(scale=1/127.5, offset=-1.0)` layer maps $[0, 255]$ pixel values directly into $[-1, 1]$, guaranteeing uniform preprocessing between training, testing, and production inference.
3. **Base Network:** Pretrained **MobileNetV2** base (`mobilenetv2_1.00_224`), initially frozen and subsequently fine-tuned on brain MRI data.
4. **Classification Head:** Global Average Pooling 2D $\rightarrow$ Dropout ($0.3$) $\rightarrow$ Dense ($4$ units, Softmax activation).
5. **Grad-CAM Target Layer:** Gradients are computed with respect to the `out_relu` activation tensor (spatial dimensions $7 \times 7 \times 1280$) using TensorFlow `tf.GradientTape()`, globally pooled, linearly combined, ReLU-filtered, and resized with bicubic interpolation into a $224 \times 224$ transparent Jet colormap overlay.

---

## 📊 Model Evaluation & Benchmarks

The model was evaluated on an untouched held-out test split of **1,195 images**:

- **Overall Test Accuracy:** **94.39%** ($1,128 / 1,195$ correct predictions)
- **Macro Average F1-Score:** **0.9431**
- **Weighted Average F1-Score:** **0.9431**

### Per-Class Test Performance

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Glioma** | 0.9580 | 0.9481 | 0.9530 | 289 |
| **Meningioma** | 0.9883 | 0.8411 | 0.9088 | 302 |
| **No tumor** | 0.9421 | 1.0000 | 0.9702 | 309 |
| **Pituitary** | 0.8981 | 0.9864 | 0.9402 | 295 |
| **Macro Average** | **0.9466** | **0.9439** | **0.9431** | **1,195** |
| **Weighted Average** | **0.9468** | **0.9439** | **0.9431** | **1,195** |

> **Evaluation Methodology Note:** Evaluation was performed using an image-level split; patient-level independence has not been established. Performance on external clinical scanners or different hospital protocols may vary.

---

## 🛠️ Technology Stack

- **Deep Learning & Inference:** TensorFlow 2.15.1, Keras 2.15.0
- **Model Evaluation:** Scikit-learn 1.9.1
- **Image Processing & Math:** Pillow 13.0.0, NumPy 1.26.4, h5py 3.16.0
- **Web Backend:** Flask 3.1.3, Werkzeug 3.1.9
- **Frontend & Visualization:** HTML5, CSS3, JavaScript (ES6+), jQuery, FontAwesome

---

## 📁 Project Structure

```text
BrainTumor-Intelligence/
├── app.py                             # Flask application entry point & inference API
├── best_brain_tumor_mobilenetv2.h5    # Trained MobileNetV2 model weights (required)
├── train_model.py                     # Model training & test evaluation script
├── requirements.txt                   # Production Python dependencies
├── .gitignore                         # Git exclusion rules
├── templates/
│   ├── base.html                      # Base HTML layout
│   └── index.html                     # NeuroScan interactive UI
├── static/
│   ├── css/
│   │   └── main.css                   # Custom application styles
│   ├── js/
│   │   └── main.js                    # Client-side upload, AJAX, & UI controller
│   └── samples/                       # Pre-loaded test MRI slices
│       ├── glioma.jpg
│       ├── meningioma.jpg
│       ├── pituitary.jpg
│       ├── no_tumor.jpg
│       └── hard_case.jpg
├── uploads/                           # Runtime user uploads directory
└── img/                               # Interface screenshots & documentation assets
```

---

## 📦 Model File

The trained model checkpoint is stored in the project root:
- **`best_brain_tumor_mobilenetv2.h5`** (~$23.5\text{ MB}$)

This file must remain in the project root directory alongside `app.py`. The application validates the presence of this model at startup and will fail explicitly if the weights file cannot be located.

---

## 🚀 Installation & Local Setup

### 1. Clone the Repository
```powershell
git clone https://github.com/chauhananamika/BrainTumor-Intelligence.git
cd BrainTumor-Intelligence
```

### 2. Create and Activate a Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Run the Web Application
```powershell
python app.py
```

### 5. Access the Web Interface
Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

---

## ⚠️ Medical Disclaimer

NeuroScan is developed strictly as an **educational and research prototype**. It is **not** an FDA-approved or CE-certified medical diagnostic system. The predictions, probabilities, and Grad-CAM visual heatmaps produced by this software must **not** be used as a substitute for professional clinical diagnosis, radiologist review, or patient treatment decisions.

---

## 📜 Credits & Acknowledgements

- **Dataset:** Masoud Nickparvar, *Brain Tumor MRI Dataset*, published under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
- **Architecture:** MobileNetV2 by Google Research (Sandler et al., 2018).
