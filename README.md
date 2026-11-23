# 🧠 NeuroScan — Brain MRI Tumor Classifier

NeuroScan is an AI-powered brain MRI classification web application built using **TensorFlow, Keras, and Flask**. It analyzes 2D brain MRI images and classifies them into four categories: **Glioma, Meningioma, Pituitary Tumor, and No Tumor**.

The application also integrates **Grad-CAM (Gradient-weighted Class Activation Mapping)** to generate visual heatmaps that highlight image regions influencing the model's predictions.

## ✨ Features

- **Four-Class MRI Classification:** Classifies MRI scans into Glioma, Meningioma, Pituitary Tumor, and No Tumor.
- **MobileNetV2 Transfer Learning:** Uses an ImageNet-pretrained MobileNetV2 backbone with a custom classification head.
- **Prediction Confidence:** Displays the predicted class, confidence score, and probability distribution across all four classes.
- **Explainable AI with Grad-CAM:** Generates activation heatmaps to visualize regions influencing the model's predictions.
- **Interactive Heatmap Controls:** Toggle heatmap visibility and adjust overlay opacity.
- **Sample MRI Images:** Includes five built-in MRI samples for testing the application.
- **Flexible Image Upload:** Supports drag-and-drop, file browsing, and clipboard paste (`Ctrl+V`).
- **Supported Formats:** PNG, JPEG, and BMP images, up to 10 MB.
- **Responsive Interface:** Provides a dark-themed interface designed for desktop and mobile screens.

---

## 🔬 Model Architecture & Processing Pipeline

The application uses a transfer-learning approach based on MobileNetV2.

### Model Architecture

1. **Input:** MRI images are converted to RGB and resized to `224 × 224 × 3`.
2. **Normalization:** An integrated Keras `Rescaling` layer transforms pixel values from `[0, 255]` to `[-1, 1]`.
3. **Base Network:** ImageNet-pretrained MobileNetV2 (`mobilenetv2_1.00_224`) extracts image features.
4. **Classification Head:**
   - Global Average Pooling 2D
   - Dropout (`0.3`)
   - Dense layer with 4 units and Softmax activation
5. **Prediction:** The Softmax output provides class probabilities for the four categories.

### Grad-CAM Visualization

Grad-CAM is implemented using TensorFlow's `tf.GradientTape()`.

- Gradients are calculated with respect to the MobileNetV2 `out_relu` convolutional activation.
- The gradients are globally pooled and combined with the activation maps.
- ReLU filtering highlights positively contributing regions.
- The resulting heatmap is resized to the input image dimensions and overlaid on the MRI scan.

This visualization provides an interpretation of the model's image-level activations. It does not independently establish the presence or location of a tumor.

---

## 📊 Model Evaluation & Benchmarks

The model was evaluated on a held-out test split containing **1,195 MRI images**.

| Metric | Result |
|---|---:|
| Test Accuracy | **94.39%** |
| Correct Predictions | 1,128 / 1,195 |
| Macro Average F1-Score | **0.9431** |
| Weighted Average F1-Score | **0.9431** |

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| Glioma | 0.9580 | 0.9481 | 0.9530 | 289 |
| Meningioma | 0.9883 | 0.8411 | 0.9088 | 302 |
| No Tumor | 0.9421 | 1.0000 | 0.9702 | 309 |
| Pituitary | 0.8981 | 0.9864 | 0.9402 | 295 |
| **Macro Average** | **0.9466** | **0.9439** | **0.9431** | **1,195** |
| **Weighted Average** | **0.9468** | **0.9439** | **0.9431** | **1,195** |

**Evaluation Note:** The evaluation used an image-level train/test split. Patient-level independence has not been established, so images from the same patient may not necessarily be independent. Performance on external datasets, clinical scanners, and different imaging protocols may vary.

---

## 🛠️ Technology Stack

| Component | Technologies |
|---|---|
| Deep Learning | TensorFlow, Keras |
| Model Architecture | MobileNetV2 |
| Model Evaluation | Scikit-learn |
| Image Processing | Pillow, NumPy, h5py |
| Backend | Python, Flask, Werkzeug |
| Frontend | HTML5, CSS3, JavaScript |
| UI Libraries | jQuery, Font Awesome |
| Explainable AI | Grad-CAM, TensorFlow GradientTape |

---

## 📁 Project Structure

```text
BrainTumor-Intelligence/
│
├── app.py                              # Flask application and inference API
├── train_model.py                      # Model training and evaluation
├── best_brain_tumor_mobilenetv2.h5     # Trained model checkpoint
├── requirements.txt                    # Python dependencies
├── .gitignore
│
├── templates/
│   ├── base.html                       # Base HTML layout
│   └── index.html                      # NeuroScan interface
│
├── static/
│   ├── css/
│   │   └── main.css                    # Application styling
│   │
│   ├── js/
│   │   └── main.js                     # Frontend interactions
│   │
│   └── samples/                        # Sample MRI images
│       ├── glioma.jpg
│       ├── meningioma.jpg
│       ├── pituitary.jpg
│       ├── no_tumor.jpg
│       └── hard_case.jpg
│
└── uploads/                            # Runtime upload directory
```

---

## 📦 Trained Model

The application uses the following trained model checkpoint:

- **Filename:** `best_brain_tumor_mobilenetv2.h5`
- **Format:** Keras HDF5 model
- **Location:** Project root directory

The model file must be present alongside `app.py` for the application to load the trained network successfully.

---

## 🚀 Installation & Local Setup

### 1. Clone the Repository

```powershell
git clone https://github.com/chauhananamika/BrainTumor-Intelligence.git
cd BrainTumor-Intelligence
```

### 2. Create a Virtual Environment

```powershell
python -m venv venv
```

### 3. Activate the Environment

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 5. Run the Application

```powershell
python app.py
```

### 6. Open in Browser

Navigate to:

```text
http://127.0.0.1:5000
```

---

## ⚠️ Medical Disclaimer

NeuroScan is developed strictly as an **educational and research prototype**. It is not an FDA-approved or CE-certified medical diagnostic system.

The predictions, confidence scores, and Grad-CAM visualizations generated by this application must not be used as a substitute for professional medical diagnosis, radiologist review, or clinical treatment decisions.

---

## 📜 Credits & Acknowledgements

- **Dataset:** Masoud Nickparvar, *Brain Tumor MRI Dataset*, distributed under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
- **Model Architecture:** MobileNetV2 by Google Research, introduced by Sandler et al. (2018).

