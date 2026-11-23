"""
Train a Four-Class Brain Tumor MRI Classification Model
Model: MobileNetV2 Transfer Learning
Target Classes: Glioma, Meningioma, No tumor, Pituitary
Cleaned Dataset: Dataset/cleaned/ (Training: 5,521 images, Testing: 1,195 images)
"""

import os
import sys
import json
import time
import random
import shutil
import numpy as np

# Suppress TF oneDNN logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYTHONHASHSEED'] = '42'

# Set fixed seeds for reproducibility
random.seed(42)
np.random.seed(42)

import tensorflow as tf
tf.random.set_seed(42)

from tensorflow.keras import layers, models
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, 'Dataset', 'cleaned', 'Training')
TEST_DIR = os.path.join(BASE_DIR, 'Dataset', 'cleaned', 'Testing')
CHECKPOINT_PATH = os.path.join(BASE_DIR, 'best_brain_tumor_mobilenetv2.h5')
FINAL_MODEL_PATH = os.path.join(BASE_DIR, 'brain_tumor_multiclass.h5')
METRICS_JSON_PATH = os.path.join(BASE_DIR, 'test_evaluation_report.json')
METRICS_TXT_PATH = os.path.join(BASE_DIR, 'test_evaluation_report.txt')

# Hyperparameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42
VALIDATION_SPLIT = 0.20
INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 6

# Explicit class mapping as specified in requirements
CLASS_MAPPING = {
    'glioma': 'Glioma',
    'meningioma': 'Meningioma',
    'notumor': 'No tumor',
    'pituitary': 'Pituitary'
}

def verify_dataset_structure():
    """Verify that cleaned dataset exists and matches requirements."""
    print("=" * 70)
    print("1. VERIFYING DATASET STRUCTURE")
    print("=" * 70)
    
    if not os.path.exists(TRAIN_DIR):
        raise FileNotFoundError(f"Training directory not found at: {TRAIN_DIR}")
    if not os.path.exists(TEST_DIR):
        raise FileNotFoundError(f"Testing directory not found at: {TEST_DIR}")
        
    train_counts = {}
    test_counts = {}
    
    for c in sorted(CLASS_MAPPING.keys()):
        train_class_dir = os.path.join(TRAIN_DIR, c)
        test_class_dir = os.path.join(TEST_DIR, c)
        
        if not os.path.exists(train_class_dir):
            raise FileNotFoundError(f"Missing training folder for class: {c}")
        if not os.path.exists(test_class_dir):
            raise FileNotFoundError(f"Missing testing folder for class: {c}")
            
        train_counts[c] = len([f for f in os.listdir(train_class_dir) if not f.startswith('.')])
        test_counts[c] = len([f for f in os.listdir(test_class_dir) if not f.startswith('.')])
        
    total_train = sum(train_counts.values())
    total_test = sum(test_counts.values())
    
    print(f"Training Set ({total_train} total images):")
    for c, cnt in train_counts.items():
        print(f"  - {c} ({CLASS_MAPPING[c]}): {cnt} images")
    print(f"Testing Set ({total_test} total images):")
    for c, cnt in test_counts.items():
        print(f"  - {c} ({CLASS_MAPPING[c]}): {cnt} images")
        
    print(f"\nVerification passed: {total_train} training images, {total_test} testing images.")
    return train_counts, test_counts

def load_datasets():
    """Load train, validation, and test datasets with tf.data pipelines."""
    print("\n" + "=" * 70)
    print("2. LOADING DATASETS (20% Validation Split from Training)")
    print("=" * 70)
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        color_mode='rgb',
        label_mode='int',
        shuffle=True
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        color_mode='rgb',
        label_mode='int',
        shuffle=False
    )
    
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        color_mode='rgb',
        label_mode='int',
        shuffle=False
    )
    
    class_names = train_ds.class_names
    print("\nVerifying class indices:")
    display_names = []
    for idx, c_name in enumerate(class_names):
        display_name = CLASS_MAPPING.get(c_name, c_name)
        display_names.append(display_name)
        print(f"  Index {idx}: '{c_name}' -> '{display_name}'")
        
    assert class_names == ['glioma', 'meningioma', 'notumor', 'pituitary'], \
        f"Unexpected class order: {class_names}"
    assert display_names == ['Glioma', 'Meningioma', 'No tumor', 'Pituitary'], \
        f"Unexpected display class names: {display_names}"
    print("Class indices successfully verified against requirements.")
    
    # Configure efficient streaming with prefetching (avoid loading whole dataset into RAM)
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)
    
    return train_ds, val_ds, test_ds, class_names, display_names

def build_model():
    """Build MobileNetV2 transfer learning model with integrated preprocessing."""
    print("\n" + "=" * 70)
    print("3. BUILDING MOBILENETV2 TRANSFER LEARNING MODEL")
    print("=" * 70)
    
    # Pretrained ImageNet weights verification
    print("Loading MobileNetV2 with ImageNet pretrained weights...")
    try:
        base_model = MobileNetV2(
            input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
            include_top=False,
            weights='imagenet'
        )
        print("ImageNet weights loaded successfully!")
    except Exception as e:
        print(f"\nFATAL ERROR: Failed to load MobileNetV2 ImageNet weights: {e}", file=sys.stderr)
        print("Refusing to silently fall back to random weights per requirements.", file=sys.stderr)
        sys.exit(1)
        
    # Initially freeze the MobileNetV2 base
    base_model.trainable = False
    print(f"Base model frozen. Base layers count: {len(base_model.layers)}")
    
    # Define model architecture with integrated MobileNetV2 preprocessing
    inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name='input_layer')
    
    # MobileNetV2 expects input scaled from [0, 255] to [-1, 1]: (x / 127.5) - 1.0
    rescaling = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name='mobilenetv2_rescaling')
    x = rescaling(inputs)
    
    # Run through frozen base in inference mode
    x = base_model(x, training=False)
    
    # Global average pooling, dropout, and 4-class softmax
    x = layers.GlobalAveragePooling2D(name='global_average_pooling')(x)
    x = layers.Dropout(0.3, name='dropout')(x)
    outputs = layers.Dense(4, activation='softmax', name='classification_head')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='BrainTumor_MobileNetV2')
    
    # Compile with Adam and sparse categorical crossentropy
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    return model, base_model

def train_model(model, base_model, train_ds, val_ds):
    """Execute training pipeline with early stopping, checkpointing, and lr reduction."""
    print("\n" + "=" * 70)
    print("4. TRAINING PIPELINE")
    print("=" * 70)
    
    callbacks = [
        ModelCheckpoint(
            filepath=CHECKPOINT_PATH,
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            mode='min',
            patience=4,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            mode='min',
            factor=0.2,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    print("\n--- Phase 1: Training Classification Head (Base Frozen) ---")
    start_time = time.time()
    history_phase1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=INITIAL_EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    p1_time = time.time() - start_time
    print(f"Phase 1 completed in {p1_time:.1f}s.")
    
    # Phase 2: Fine-Tuning top layers of MobileNetV2
    print("\n--- Phase 2: Fine-Tuning Top Layers of MobileNetV2 ---")
    base_model.trainable = True
    fine_tune_at = 110  # Freeze lower 110 layers, train remaining 44 layers
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
        
    trainable_count = sum(len(layer.weights) for layer in base_model.layers if layer.trainable)
    print(f"Unfrozen top layers from layer {fine_tune_at} onwards ({trainable_count} trainable weight tensors).")
    
    # Recompile with smaller learning rate for gentle fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    ft_start = time.time()
    history_phase2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=FINE_TUNE_EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    ft_time = time.time() - ft_start
    print(f"Phase 2 completed in {ft_time:.1f}s.")
    
    # Also save/copy best checkpoint to brain_tumor_multiclass.h5 for app.py
    if os.path.exists(CHECKPOINT_PATH):
        shutil.copyfile(CHECKPOINT_PATH, FINAL_MODEL_PATH)
        print(f"\nSaved best model checkpoint to:")
        print(f"  - {CHECKPOINT_PATH}")
        print(f"  - {FINAL_MODEL_PATH}")
        
    return model

def evaluate_on_test_set(test_ds, display_names):
    """Evaluate best model on untouched testing dataset and compute all metrics."""
    print("\n" + "=" * 70)
    print("5. EVALUATION ON UNTOUCHED TEST SET")
    print("=" * 70)
    
    print(f"Loading best checkpoint from: {CHECKPOINT_PATH}")
    best_model = tf.keras.models.load_model(CHECKPOINT_PATH)
    
    print("Generating predictions on untouched testing dataset...")
    y_true_list = []
    y_pred_list = []
    y_prob_list = []
    
    for images, labels in test_ds:
        preds = best_model(images, training=False)
        y_prob_list.append(preds.numpy())
        y_pred_list.append(np.argmax(preds.numpy(), axis=1))
        y_true_list.append(labels.numpy())
        
    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)
    y_probs = np.concatenate(y_prob_list, axis=0)
    
    num_samples = len(y_true)
    print(f"Evaluated on {num_samples} untouched test images.")
    
    # 1. Overall Test Accuracy
    test_acc = accuracy_score(y_true, y_pred)
    
    # 2. Per-Class Precision, Recall, F1-Score, Support
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2, 3], average=None
    )
    
    # Macro and weighted averages
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro'
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    
    # 3. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    
    # 4. Classification Report
    clf_report = classification_report(
        y_true, y_pred,
        target_names=display_names,
        digits=4
    )
    
    # Display Results
    print("\n" + "-" * 70)
    print(f"OVERALL TEST ACCURACY: {test_acc * 100:.2f}% ({np.sum(y_true == y_pred)} / {num_samples})")
    print("-" * 70)
    
    print("\nPER-CLASS PERFORMANCE METRICS:")
    print(f"{'Class Name':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 65)
    per_class_dict = {}
    for i, name in enumerate(display_names):
        print(f"{name:<15} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<10}")
        per_class_dict[name] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1_score': float(f1[i]),
            'support': int(support[i])
        }
    print("-" * 65)
    print(f"{'Macro Avg':<15} {macro_p:<12.4f} {macro_r:<12.4f} {macro_f1:<12.4f} {num_samples:<10}")
    print(f"{'Weighted Avg':<15} {weighted_p:<12.4f} {weighted_r:<12.4f} {weighted_f1:<12.4f} {num_samples:<10}")
    
    header_row = f"{'True / Pred':<15} " + " ".join([f"{name[:8]:>9}" for name in display_names])
    print("\nCONFUSION MATRIX:")
    print(header_row)
    for i, name in enumerate(display_names):
        row_str = " ".join([f"{cm[i][j]:>9}" for j in range(4)])
        print(f"{name:<15} {row_str}")
        
    print("\nFULL CLASSIFICATION REPORT:")
    print(clf_report)
    
    # Save metrics to JSON
    metrics_data = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'model_architecture': 'MobileNetV2 (ImageNet Pretrained)',
        'input_shape': [IMG_SIZE[0], IMG_SIZE[1], 3],
        'total_test_samples': int(num_samples),
        'test_accuracy': float(test_acc),
        'per_class_metrics': per_class_dict,
        'macro_averages': {
            'precision': float(macro_p),
            'recall': float(macro_r),
            'f1_score': float(macro_f1)
        },
        'weighted_averages': {
            'precision': float(weighted_p),
            'recall': float(weighted_r),
            'f1_score': float(weighted_f1)
        },
        'confusion_matrix': cm.tolist(),
        'class_order': display_names
    }
    
    with open(METRICS_JSON_PATH, 'w') as f:
        json.dump(metrics_data, f, indent=4)
    print(f"\nSaved metrics JSON to: {METRICS_JSON_PATH}")
    
    # Save text report
    with open(METRICS_TXT_PATH, 'w') as f:
        f.write("BRAIN TUMOR MRI FOUR-CLASS CLASSIFIER - TEST EVALUATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Model: MobileNetV2 (ImageNet Pretrained Transfer Learning)\n")
        f.write(f"Total Test Samples: {num_samples}\n")
        f.write(f"Overall Test Accuracy: {test_acc * 100:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(clf_report + "\n\n")
        f.write("Confusion Matrix:\n")
        f.write(f"{'True / Pred':<15} " + " ".join([f"{name[:8]:>9}" for name in display_names]) + "\n")
        for i, name in enumerate(display_names):
            row_str = " ".join([f"{cm[i][j]:>9}" for j in range(4)])
            f.write(f"{name:<15} {row_str}\n")
    print(f"Saved text report to: {METRICS_TXT_PATH}")
    
    return metrics_data

def main():
    print("\n" + "=" * 70)
    print("STARTING BRAIN TUMOR FOUR-CLASS MRI CLASSIFIER TRAINING PIPELINE")
    print("=" * 70)
    total_start = time.time()
    
    # Step 1: Verify Dataset
    verify_dataset_structure()
    
    # Step 2: Load Data
    train_ds, val_ds, test_ds, class_names, display_names = load_datasets()
    
    # Step 3: Build Model
    model, base_model = build_model()
    
    # Step 4: Train Model
    trained_model = train_model(model, base_model, train_ds, val_ds)
    
    # Step 5: Evaluate Model on Untouched Test Set
    metrics = evaluate_on_test_set(test_ds, display_names)
    
    total_time = time.time() - total_start
    print("\n" + "=" * 70)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {total_time/60:.2f} MINUTES")
    print(f"FINAL TEST ACCURACY: {metrics['test_accuracy'] * 100:.2f}%")
    print("=" * 70)

if __name__ == '__main__':
    main()
