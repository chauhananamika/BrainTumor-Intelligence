import os
import sys
import time
import io
import base64
import numpy as np
from PIL import Image

# Suppress TF oneDNN warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILENAME = 'best_brain_tumor_mobilenetv2.h5'
MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILENAME)
if not os.path.exists(MODEL_PATH):
    fallback_path = os.path.join(BASE_DIR, 'brain_tumor_multiclass.h5')
    if os.path.exists(fallback_path):
        MODEL_PATH = fallback_path

UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
SAMPLES_DIR = os.path.join(BASE_DIR, 'static', 'samples')
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)

# Exact 4-class mapping verified from dataset and model training:
# 0 -> Glioma, 1 -> Meningioma, 2 -> No tumor, 3 -> Pituitary
CLASSES = ['Glioma', 'Meningioma', 'No tumor', 'Pituitary']

def load_verified_model():
    """Load the verified MobileNetV2 model at startup. Fail clearly if missing/corrupt."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Verified model not found at {MODEL_PATH}. "
            "Refusing to run with unverified or randomly initialized model."
        )
    print(f"Loading verified MobileNetV2 model from: {MODEL_PATH}")
    try:
        loaded_model = load_model(MODEL_PATH)
        print("MobileNetV2 model loaded successfully.")
        return loaded_model
    except Exception as e:
        raise RuntimeError(f"Failed to load verified model from {MODEL_PATH}: {e}")

# Load full model at application startup
raw_model = load_verified_model()

def build_gradcam_model(trained_model):
    """
    Build a gradient model connecting:
      Input (224, 224, 3) -> Rescaling -> MobileNetV2 Base (out_relu) -> GAP -> Dense Head
    Returns outputs: [conv_features, pre_softmax_logits, softmax_probabilities]
    """
    print("Constructing Grad-CAM gradient model targeting 'out_relu'...")
    base_layer = trained_model.get_layer('mobilenetv2_1.00_224')
    target_conv = base_layer.get_layer('out_relu')
    
    # Sub-model for the base that outputs the target convolutional activation tensor
    base_conv_model = tf.keras.Model(
        inputs=base_layer.inputs,
        outputs=target_conv.output,
        name='base_conv_submodel'
    )
    
    inputs = tf.keras.Input(shape=(224, 224, 3), name='gradcam_input')
    x = trained_model.get_layer('mobilenetv2_rescaling')(inputs)
    conv_features = base_conv_model(x, training=False)
    
    x_pooled = trained_model.get_layer('global_average_pooling')(conv_features)
    
    # Dense classification head
    dense_layer = trained_model.get_layer('classification_head')
    weights, bias = dense_layer.get_weights()
    logits = tf.matmul(x_pooled, weights) + bias
    preds = tf.nn.softmax(logits)
    
    gradcam_model = tf.keras.Model(
        inputs=inputs,
        outputs=[conv_features, logits, preds],
        name='MobileNetV2_GradCAM'
    )
    print("Grad-CAM model constructed successfully.")
    return gradcam_model

grad_model = build_gradcam_model(raw_model)

def preprocess_image(pil_img):
    """
    Shared image preprocessing pipeline:
    - Convert input to RGB
    - Resize to 224 x 224
    - Convert to float32 NumPy array with batch dimension (1, 224, 224, 3)
    - Keep pixel values in [0, 255] (model has an internal Rescaling layer)
    """
    img_rgb = pil_img.convert('RGB')
    img_resized = img_rgb.resize((224, 224), Image.Resampling.BILINEAR)
    img_array = np.array(img_resized, dtype=np.float32)
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch

def generate_gradcam_overlay(conv_features, grads):
    """
    Mathematical Grad-CAM algorithm:
    1. Channel-wise global average pooling of gradients: alpha_k = (1/Z) * sum_{i,j} dY_c / dA^k_{i,j}
    2. Linear combination of feature maps: L_c = sum_k alpha_k * A^k
    3. Apply ReLU: only positive contributions to class c are preserved
    4. Normalize to [0, 1]
    5. Resize to 224 x 224 with bicubic interpolation
    6. Apply Jet colormap with alpha transparency
    Returns: Data URI string 'data:image/png;base64,...'
    """
    # 1. Global average pooling of gradients over spatial dimensions (height, width)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # Shape: (1280,)
    
    # 2. Weighted sum of feature map channels
    heatmap = tf.reduce_sum(conv_features[0] * pooled_grads, axis=-1)  # Shape: (7, 7)
    
    # 3. Apply ReLU
    heatmap = tf.maximum(heatmap, 0.0)
    
    # 4. Normalize
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0.0:
        heatmap = heatmap / max_val
    hm_7x7 = heatmap.numpy()
    
    # 5. Resize to 224 x 224 using bicubic interpolation
    hm_pil = Image.fromarray((hm_7x7 * 255.0).astype(np.uint8), mode='L')
    hm_224 = hm_pil.resize((224, 224), Image.Resampling.BICUBIC)
    hm_arr = np.array(hm_224, dtype=np.float32) / 255.0
    
    # 6. Jet colormap formula in NumPy
    r = np.clip(1.5 - np.abs(hm_arr * 4.0 - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(hm_arr * 4.0 - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(hm_arr * 4.0 - 1.0), 0.0, 1.0)
    
    # Alpha channel: transparent for low activations (< 0.12), smoothly opaque for salient features
    alpha = np.clip((hm_arr - 0.12) / 0.88 * 255.0, 0.0, 255.0)
    
    rgba = np.zeros((224, 224, 4), dtype=np.uint8)
    rgba[..., 0] = (r * 255.0).astype(np.uint8)
    rgba[..., 1] = (g * 255.0).astype(np.uint8)
    rgba[..., 2] = (b * 255.0).astype(np.uint8)
    rgba[..., 3] = alpha.astype(np.uint8)
    
    rgba_img = Image.fromarray(rgba, 'RGBA')
    buf = io.BytesIO()
    rgba_img.save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('utf-8')

def predict_mri(pil_img, compute_cam=True):
    """
    Genuine MobileNetV2 inference and Grad-CAM:
    Runs inference via grad_model with tf.GradientTape.
    Returns:
        predicted_class (str): Name of winning class.
        confidence (float): Winning softmax probability percentage (0.0 to 100.0).
        probabilities (dict): Dictionary mapping class names to percentage probabilities.
        heatmap_base64 (str or None): Genuine Grad-CAM Base64 PNG or None if failed.
    """
    img_batch = preprocess_image(pil_img)
    
    with tf.GradientTape() as tape:
        conv_features, logits, preds_tensor = grad_model(img_batch, training=False)
        pred_idx = int(tf.argmax(preds_tensor[0]).numpy())
        class_score = logits[:, pred_idx]
        
    preds = preds_tensor.numpy()[0]
    predicted_class = CLASSES[pred_idx]
    confidence = float(preds[pred_idx] * 100.0)
    
    probabilities = {
        CLASSES[i]: round(float(preds[i] * 100.0), 2)
        for i in range(len(CLASSES))
    }
    
    heatmap_base64 = None
    if compute_cam:
        try:
            grads = tape.gradient(class_score, conv_features)
            if grads is not None:
                heatmap_base64 = generate_gradcam_overlay(conv_features, grads)
        except Exception as e:
            print(f"Warning: Grad-CAM generation failed: {e}", file=sys.stderr)
            heatmap_base64 = None
            
    return predicted_class, confidence, probabilities, heatmap_base64

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict_sample/<sample_name>', methods=['GET'])
def predict_sample(sample_name):
    t0 = time.time()
    sample_filename = f"{sample_name}.jpg"
    sample_img_path = os.path.join(SAMPLES_DIR, sample_filename)

    if not os.path.exists(sample_img_path):
        return jsonify({'error': f'Sample scan "{sample_filename}" not found.'}), 404

    try:
        with Image.open(sample_img_path) as img:
            pil_img = img.copy()
            pil_img.load()
    except Exception as e:
        return jsonify({'error': f'Failed to decode sample image: {e}'}), 500

    predicted_class, confidence, probabilities, heatmap_b64 = predict_mri(pil_img, compute_cam=True)
    latency_ms = max(int((time.time() - t0) * 1000), 1)

    return jsonify({
        'prediction': predicted_class,
        'confidence': round(confidence, 1),
        'probabilities': probabilities,
        'latency_ms': latency_ms,
        'distribution_check': 'Not assessed',
        'heatmap_base64': heatmap_b64,
        'image_url': f"/static/samples/{sample_name}.jpg"
    })

@app.route('/predict', methods=['POST'])
def upload():
    t0 = time.time()
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    f = request.files['file']
    if not f or f.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filename = secure_filename(f.filename)
    if not filename:
        filename = f"upload_{int(time.time() * 1000)}.jpg"
    file_path = os.path.join(UPLOAD_DIR, filename)
    f.save(file_path)

    try:
        with Image.open(file_path) as img:
            pil_img = img.copy()
            pil_img.load()
    except Exception as e:
        return jsonify({'error': f'Invalid image format: {e}'}), 400

    predicted_class, confidence, probabilities, heatmap_b64 = predict_mri(pil_img, compute_cam=True)
    latency_ms = max(int((time.time() - t0) * 1000), 1)

    return jsonify({
        'prediction': predicted_class,
        'confidence': round(confidence, 1),
        'probabilities': probabilities,
        'latency_ms': latency_ms,
        'distribution_check': 'Not assessed',
        'heatmap_base64': heatmap_b64
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"NeuroScan web server running at: http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
