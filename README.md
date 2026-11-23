# 🧠 NeuroScan — Brain MRI Tumor Classifier

NeuroScan is an AI-powered brain MRI classification web application built using **TensorFlow, Keras, and Flask**. It analyzes 2D brain MRI images and classifies them into four categories: **Glioma, Meningioma, Pituitary Tumor, and No Tumor**.

The application also integrates **Grad-CAM (Gradient-weighted Class Activation Mapping)** to generate visual heatmaps that highlight image regions influencing the model's predictions.

## ✨ Features

- **Four-Class MRI Classification:** Classifies MRI scans into Glioma, Meningioma, Pituitary Tumor, and No Tumor.
- **MobileNetV2 Transfer Learning:** Uses an ImageNet-pretrained MobileNetV2 backbone with a custom classification head.
- **Prediction Confidence:** Displays the predicted class, confidence score, and probability distribution across all four classes.
- **Explainable AI with Grad-CAM:** Generates activation heatmaps to visualize
