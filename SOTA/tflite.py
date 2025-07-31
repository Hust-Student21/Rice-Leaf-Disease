import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import os
import shutil

# === Paths ===
onnx_model_path = r"D:\STUDY_HARD\Uni\2024.2\THESIS\Rice-Plant-Disease-Detection\runs\train\ema_org\weights\gema.onnx"
saved_model_dir = r"D:\STUDY_HARD\Uni\2024.2\THESIS\Rice-Plant-Disease-Detection\runs\train\ema_org\weights\model_tf"
tflite_model_path = r"D:\STUDY_HARD\Uni\2024.2\THESIS\Rice-Plant-Disease-Detection\runs\train\ema_org\weights\gema.tflite"

# === Clean previous TF output ===
if os.path.exists(saved_model_dir):
    shutil.rmtree(saved_model_dir)

# === Step 1: Load ONNX and convert to TensorFlow ===
print("🔁 Loading ONNX model...")
onnx_model = onnx.load(onnx_model_path)

print("🔁 Converting to TensorFlow...")
tf_rep = prepare(onnx_model)
tf_rep.export_graph(saved_model_dir)

print(f"✅ TensorFlow SavedModel saved at: {saved_model_dir}")

# === Step 2: Convert to TFLite ===
print("✨ Converting to TFLite...")

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save the TFLite model
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print(f"✅ TFLite model saved at: {tflite_model_path}")
