from ultralytics import YOLO

def count_params(model_path):
    model = YOLO(model_path)
    return model.info(verbose=True)

# Gọi thử:
count_params(r'yolov8n.pt')  # hoặc 'runs/detect/train/weights/best.pt'
count_params(r'D:\STUDY_HARD\Uni\2024.2\THESIS\Rice-Plant-Disease-Detection\runs\train\100e_20_3\weights\100e203.pt')

import onnx

import torch

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🔢 Total parameters: {total:,}")
    print(f"🧠 Trainable parameters: {trainable:,}")
    return total, trainable

# Load model manually
model = torch.load(r'D:\STUDY_HARD\Uni\2024.2\THESIS\Rice-Plant-Disease-Detection\100e203_pruned.pt')
count_parameters(model)
def count_onnx_params(onnx_path):
    model = onnx.load(onnx_path)
    total_params = 0
    for tensor in model.graph.initializer:
        dims = 1
        for dim in tensor.dims:
            dims *= dim
        total_params += dims
    print(f"📦 ONNX Model: {onnx_path}")
    print(f"🔢 Total Parameters: {total_params:,}")

# Gọi thử:
count_onnx_params(r'D:\STUDY_HARD\Uni\2024.2\THESIS\Rice-Plant-Disease-Detection\runs\train\100e_20_3\weights\100e203.onnx')
count_onnx_params(r'D:\STUDY_HARD\Uni\2024.2\THESIS\Rice-Plant-Disease-Detection\100e203_quantitized.onnx')
count_onnx_params(r'D:\STUDY_HARD\Uni\2024.2\THESIS\Rice-Plant-Disease-Detection\100e203_fp16.onnx')
