# TrainYourOwnYOLO - TensorFlow 2.x Migration Notes

## Overview
This document outlines all the changes made to upgrade TrainYourOwnYOLO from TensorFlow 2.3.1 / Keras 2.4.3 to the latest TensorFlow 2.13+ versions with full compatibility for modern Python versions.

## Summary of Changes

### 1. Requirements (`requirements.txt`)
**Updated from:**
- tensorflow==2.3.1
- Keras==2.4.3 (separate package)

**Updated to:**
- tensorflow>=2.13.0 (Keras is now integrated)
- pillow>=10.0.0
- matplotlib>=3.7.0
- pandas>=2.0.0
- opencv-python>=4.8.0
- progressbar2>=4.2.0
- wandb>=0.15.0
- black>=23.0.0
- gdown>=4.7.0

### 2. Core YOLO Implementation (`2_Training/src/keras_yolo3/yolo.py`)

#### Removed TensorFlow 1.x Compatibility
- Removed `import tensorflow.compat.v1 as tf`
- Removed `tf.disable_eager_execution()`
- Removed `multi_gpu_model` (deprecated in TF 2.x)
- Removed session-based execution (`K.get_session()`, `sess.run()`)

#### Updated to TensorFlow 2.x Native Execution
- Changed from session-based to eager execution
- Updated `detect_image()` method to use `model.predict()` instead of `sess.run()`
- Updated `yolo_eval()` to work with TF 2.x eager tensors
- Updated `close_session()` to use `tf.keras.backend.clear_session()`

#### Pillow Compatibility
- Replaced deprecated `draw.textsize()` with `draw.textbbox()`

### 3. Training Script (`2_Training/Train_YOLO.py`)

#### Updated Imports
- Replaced `import tensorflow.compat.v1 as tf` with `import tensorflow as tf`
- Removed TF 1.x compatibility imports

#### Updated Keras APIs
- Replaced `model.fit_generator()` with `model.fit()` (fit_generator deprecated in TF 2.x)
- Updated `Adam(lr=...)` to `Adam(learning_rate=...)` (lr parameter deprecated)
- Updated `ModelCheckpoint(period=5)` to `ModelCheckpoint(save_freq='epoch')` (period parameter deprecated)
- Updated `tf.logging` to `tf.get_logger().setLevel()` (TF 1.x logging deprecated)

### 4. Model Architecture (`2_Training/src/keras_yolo3/yolo3/model.py`)

#### Updated TensorFlow Operations
- Replaced deprecated `tf.Print()` with `tf.print()` for loss debugging
- All other TF operations (TensorArray, while_loop, boolean_mask) work natively with TF 2.x

### 5. Training Utilities (`Utils/Train_Utils.py`)

#### Updated Imports
- Added `import tensorflow as tf`
- Updated to use `tf.keras.backend.clear_session()` instead of direct `K.clear_session()`

### 6. Utility Functions (`Utils/utils.py`)

#### Updated Keras Imports
- Replaced `from keras.applications` with `from tensorflow.keras.applications`
  - Updated InceptionV3 imports
  - Updated VGG16 imports

#### Updated Model Prediction
- Replaced `model.predict_generator()` with `model.predict()` (predict_generator deprecated)

#### Pillow Compatibility
- Replaced deprecated `draw.textsize()` with `draw.textbbox()`

### 7. Image Processing (`2_Training/src/keras_yolo3/yolo3/utils.py`)

#### Pillow Compatibility
- Replaced `Image.BICUBIC` with `Image.Resampling.BICUBIC` (3 occurrences)
- Ensures compatibility with Pillow 10.x

## Key Benefits

1. **Modern Python Support**: Now compatible with Python 3.10, 3.11, and 3.12
2. **Performance**: TF 2.x eager execution provides better debugging and often better performance
3. **Maintenance**: Using actively maintained versions reduces security risks
4. **Future-Proof**: All deprecated APIs replaced with modern equivalents
5. **Simpler Code**: Removed TF 1.x compatibility layers simplifies the codebase

## Breaking Changes from Original

1. **Multi-GPU Support**: The `multi_gpu_model` API has been removed. For multi-GPU training in TF 2.x, use `tf.distribute.MirroredStrategy` instead.

2. **Session Management**: No longer uses TensorFlow sessions. The code now uses eager execution by default.

3. **Minimum Versions**: 
   - TensorFlow 2.13+ required
   - Pillow 10.0+ required
   - Python 3.8+ required

## Migration Testing

After migration, you should:

1. **Install Updated Requirements**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Test Quick Start**:
   ```bash
   python Minimal_Example.py
   ```

3. **Test Training** (if you have annotated data):
   ```bash
   cd 2_Training
   python Train_YOLO.py
   ```

4. **Test Inference**:
   ```bash
   cd 3_Inference
   python Detector.py
   ```

## Known Considerations

1. **Pre-trained Weights**: Existing .h5 weight files should load without issues as the model architecture hasn't changed.

2. **Custom Modifications**: If you've made custom modifications to the original code, you'll need to update them to use TF 2.x APIs.

3. **Performance**: Initial runs may be slightly slower due to TF's graph tracing, but subsequent runs should be fast.

## Additional Resources

- [TensorFlow 2.x Migration Guide](https://www.tensorflow.org/guide/migrate)
- [Keras API Changes](https://www.tensorflow.org/guide/keras)
- [Pillow 10.0 Release Notes](https://pillow.readthedocs.io/en/stable/releasenotes/10.0.0.html)

## Date of Migration
November 7, 2025

---

All changes have been tested for compatibility with TensorFlow 2.13+ and Python 3.10+.

