# Bird Detector — Keras/TensorFlow (MobileNetV3-Large)

Binary bird presence detector for a fixed balcony camera. Classifies each frame as **Bird (1)** or **No-Bird (0)**. Exported to ONNX for Android deployment.

---

## Datasets Required

Add these three datasets to the notebook before running:

| Dataset | Kaggle Slug | Path in Notebook |
|---|---|---|
| Custom balcony images | `sreekanthnanduri/balcony-bird-dataset` | `/kaggle/input/datasets/sreekanthnanduri/balcony-bird-dataset/balcony-bird-dataset/` |
| Birdies (general birds) | `gpiosenka/birdies` | `/kaggle/input/datasets/gpiosenka/birdies/images/` |
| COCO 2017 (no-bird class) | `awsaf49/coco-2017-dataset` | `/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017/` |

---

## How to Run

1. Open this notebook on Kaggle
2. Add the three datasets above via **Add Data**
3. Enable **GPU** (Settings → Accelerator → GPU T4 x2)
4. Run all cells top to bottom

---

## Input Paths (CONFIG)

```python
'custom_birds_dir':  '/kaggle/input/datasets/sreekanthnanduri/balcony-bird-dataset/balcony-bird-dataset/birds'
'custom_bg_dir':     '/kaggle/input/datasets/sreekanthnanduri/balcony-bird-dataset/balcony-bird-dataset/background'
'kaggle_birds_dir':  '/kaggle/input/datasets/gpiosenka/birdies/images'
'coco_img_dir':      '/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017/train2017'
'coco_ann_file':     '/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017/annotations/instances_train2017.json'
```

---

## Output Files

All saved to `/kaggle/working/`:

| File | Description |
|---|---|
| `models/bird_detector_keras.onnx` | Android deployment via ONNX Runtime |
| `models/bird_detector_keras.h5` | HDF5 format — classic Keras, widely compatible |
| `models/bird_detector_keras.keras` | Native Keras format — recommended for TF2 fine-tuning |
| `models/best_phase1.keras` | Best checkpoint from Phase 1 |
| `models/best_phase2.keras` | Best checkpoint from Phase 2 |
| `keras_training_curves.png` | Loss + Accuracy curves (both phases) |
| `keras_confusion_matrix.png` | Confusion matrix on test set |
| `keras_roc_curve.png` | ROC curve + AUC score |
| `keras_pr_curve.png` | Precision-Recall curve |
| `keras_sample_predictions.png` | 16 sample predictions grid |

---

## Model Details

| Property | Value |
|---|---|
| Architecture | MobileNetV3-Large |
| Pretrained on | ImageNet |
| Input size | 224×224×3 NHWC (BILINEAR resize, no cropping) |
| Output | Sigmoid probability → threshold 0.5 |
| ONNX input format | NHWC `(1, 224, 224, 3)` |
| Loss | BinaryCrossentropy |
| Optimizer | Adam (weight_decay=1e-4) |
| LR Scheduler | ReduceLROnPlateau |

---

## Training Phases

**Phase 1 — Head only (backbone frozen)**
- Epochs: 5
- LR: 1e-3
- Trains only the `Dense(1, sigmoid)` head

**Phase 2 — Full fine-tune (all layers unfrozen)**
- Epochs: 20
- LR: 1e-4
- Fine-tunes entire network

---

## Preprocessing & Augmentation

- Resize to 224×224 using **BILINEAR** interpolation (no cropping)
- Normalize with ImageNet mean/std: `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`
- Augmentation applied to custom images only (5x per image, saved to disk):
  - Horizontal flip, brightness/contrast jitter, Gaussian blur
  - Rotation ±15°, hue/saturation shift, Gaussian noise

---

## Load Trained Model

```python
import tensorflow as tf

# Option 1 — Native Keras format (recommended)
model = tf.keras.models.load_model('bird_detector_keras.keras')

# Option 2 — HDF5 format
model = tf.keras.models.load_model('bird_detector_keras.h5')
```

## Run ONNX Inference

```python
import onnxruntime as ort
import numpy as np
from PIL import Image

session = ort.InferenceSession('bird_detector_keras.onnx')

img = Image.open('your_image.jpg').convert('RGB').resize((224, 224))
img = np.array(img).astype(np.float32) / 255.0
img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
img = img[np.newaxis]  # NHWC: (1, 224, 224, 3)

prob = session.run(None, {'input': img})[0][0][0]
print('Bird' if prob >= 0.5 else 'No Bird', f'({prob:.2f})')
```

---

## Note on ONNX Input Format

| Framework | Format | Shape |
|---|---|---|
| PyTorch ONNX | NCHW | `(1, 3, 224, 224)` |
| Keras ONNX | NHWC | `(1, 224, 224, 3)` |

Use the correct format when integrating into the Android app.
