import torch
import os
from ultralytics import YOLO
from thop import profile
from thop.vision.basic_hooks import count_convNd, count_bn, count_linear

def get_model_info(model, input_size=(3, 640, 640)):
    dummy_input = torch.randn(1, *input_size)

    # Remove Detect layer if thop breaks on it
    for m in model.modules():
        if m.__class__.__name__ == 'Detect':
            m.forward = lambda x: x  # patch Detect temporarily

    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    size_mb = sum(p.element_size() * p.nelement() for p in model.parameters()) / 1024**2
    return params, flops, size_mb

def scan_yolov8_models(folder_path):
    model_files = [f for f in os.listdir(folder_path) if f.endswith('.pt')]
    if not model_files:
        print("❌ No .pt model files found.")
        return

    print(f"{'Model Name':<25} {'Params (M)':<12} {'FLOPs (GFLOPs)':<15} {'Size (MB)':<12}")
    print("-" * 70)

    for file in model_files:
        try:
            model_path = os.path.join(folder_path, file)
            yolo = YOLO(model_path)
            core_model = yolo.model  # actual nn.Module
            core_model.eval()
            params, flops, size_mb = get_model_info(core_model, input_size=(3, 640, 640))
            print(f"{file:<25} {params / 1e6:<12.2f} {flops / 1e9:<15.2f} {size_mb:<12.2f}")
        except Exception as e:
            print(f"{file:<25} Failed: {e}")

# 🔍 Set your model directory here
if __name__ == "__main__":
    folder_path = r"D:\STUDY_HARD\Uni\2024.2\THESIS\Rice-Plant-Disease-Detection\Models_Check"
    scan_yolov8_models(folder_path)
