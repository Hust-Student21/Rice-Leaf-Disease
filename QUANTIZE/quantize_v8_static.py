import os
import cv2
import numpy as np
from onnxruntime.quantization import quantize_static, CalibrationDataReader
from onnxruntime.quantization import QuantFormat, QuantType
from pathlib import Path

# ========== CONFIG ================
ONNX_MODEL_PATH = r"D:\STUDY_HARD\Uni\2024.2\THESIS\Rice-Plant-Disease-Detection\LSKA\Thesis.onnx"
QUANTIZED_MODEL_PATH = r"custom_yolov8_int8_static.onnx"
CALIBRATION_IMAGE_DIR = r"D:\STUDY_HARD\Uni\2024.2\THESIS\Rice-Plant-Disease-Detection\QUANTIZE\calib"  # Folder with calibration images
INPUT_SIZE = 640
N_IMAGES = 50  # Number of images to use for calibration
INPUT_NAME = "images"  # Your model input name (may vary)
# ==================================

# --- Step 1: Preprocessing ---
def preprocess_yolov8(img_path, input_size=640):
    img = cv2.imread(str(img_path))
    img = cv2.resize(img, (input_size, input_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0  # normalize to [0, 1]
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, 0).astype(np.float32)  # NCHW
    return img

# --- Step 2: Calibration Data Reader ---
class YoloCalibData(CalibrationDataReader):
    def __init__(self, image_dir, n_images=10):
        image_paths = list(Path(image_dir).glob("*.[jp]*g"))[:n_images]
        self.images = iter([preprocess_yolov8(p) for p in image_paths])

    def get_next(self):
        try:
            image = next(self.images)
            return {INPUT_NAME: image}
        except StopIteration:
            return None

# --- Step 3: Run Static Quantization ---
def run_quantization():
    print("Starting quantization...")

    data_reader = YoloCalibData(CALIBRATION_IMAGE_DIR, N_IMAGES)

    quantize_static(
        model_input=ONNX_MODEL_PATH,
        model_output=QUANTIZED_MODEL_PATH,
        calibration_data_reader=data_reader,
        quant_format=QuantFormat.QOperator,   # More compatible for deployment
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8
    )

    print(f"Quantized model saved to: {QUANTIZED_MODEL_PATH}")

# --- Run ---
if __name__ == "__main__":
    run_quantization()
