#!/usr/bin/env python3
"""
Test script to verify TensorFlow 2.x migration was successful.
Run this after installing requirements to ensure all imports work correctly.
"""

import sys


def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    errors = []

    # Test TensorFlow
    try:
        import tensorflow as tf

        print(f"✓ TensorFlow {tf.__version__} imported successfully")
        if tf.__version__.startswith("2."):
            print(f"  ✓ TensorFlow 2.x detected")
        else:
            errors.append(
                f"  ✗ TensorFlow version {tf.__version__} may not be compatible"
            )
    except ImportError as e:
        errors.append(f"✗ Failed to import TensorFlow: {e}")

    # Test Keras (integrated in TF 2.x)
    try:
        from tensorflow import keras

        print(f"✓ Keras (from TensorFlow) imported successfully")
    except ImportError as e:
        errors.append(f"✗ Failed to import Keras: {e}")

    # Test Pillow
    try:
        from PIL import Image
        import PIL

        print(f"✓ Pillow {PIL.__version__} imported successfully")

        # Check for Image.Resampling (required for Pillow 10+)
        if hasattr(Image, "Resampling"):
            print(f"  ✓ Pillow 10+ API detected")
        else:
            errors.append(f"  ✗ Pillow version may be too old (need 10.0+)")
    except ImportError as e:
        errors.append(f"✗ Failed to import Pillow: {e}")

    # Test other dependencies
    packages = [
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("cv2", "OpenCV"),
        ("matplotlib", "Matplotlib"),
        ("progressbar2", "ProgressBar2"),
    ]

    for module, name in packages:
        try:
            __import__(module)
            print(f"✓ {name} imported successfully")
        except ImportError as e:
            errors.append(f"✗ Failed to import {name}: {e}")

    # Optional: wandb
    try:
        import wandb

        print(f"✓ Weights & Biases imported successfully (optional)")
    except ImportError:
        print(f"⚠ Weights & Biases not installed (optional, not an error)")

    return errors


def test_tf_features():
    """Test TensorFlow 2.x features."""
    print("\nTesting TensorFlow 2.x features...")
    errors = []

    try:
        import tensorflow as tf

        # Test eager execution (should be enabled by default in TF 2.x)
        if tf.executing_eagerly():
            print("✓ Eager execution is enabled (TF 2.x default)")
        else:
            errors.append("✗ Eager execution is not enabled")

        # Test basic tensor operation
        a = tf.constant([[1, 2], [3, 4]])
        b = tf.constant([[5, 6], [7, 8]])
        c = tf.matmul(a, b)
        print("✓ Basic tensor operations work")

        # Test Keras model creation
        from tensorflow.keras import layers, models

        model = models.Sequential(
            [
                layers.Dense(10, activation="relu", input_shape=(5,)),
                layers.Dense(2, activation="softmax"),
            ]
        )
        print("✓ Keras model creation works")

    except Exception as e:
        errors.append(f"✗ TensorFlow feature test failed: {e}")

    return errors


def main():
    """Run all tests."""
    print("=" * 60)
    print("TrainYourOwnYOLO - Installation Verification")
    print("=" * 60)

    print(f"\nPython version: {sys.version}")

    # Run import tests
    import_errors = test_imports()

    # Run TF feature tests
    feature_errors = test_tf_features()

    # Summary
    print("\n" + "=" * 60)
    all_errors = import_errors + feature_errors

    if not all_errors:
        print("✓ All tests passed! Installation is successful.")
        print("\nYou can now:")
        print("  1. Run the minimal example: python Minimal_Example.py")
        print("  2. Train your model: cd 2_Training && python Train_YOLO.py")
        print("  3. Run inference: cd 3_Inference && python Detector.py")
        return 0
    else:
        print(f"✗ {len(all_errors)} error(s) found:")
        for error in all_errors:
            print(f"  {error}")
        print("\nPlease install missing packages:")
        print("  pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())
