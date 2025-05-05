
import os

# Danh sách các file model muốn kiểm tra
model_files = {
    "YOLOv8 Gốc (.pt)": r'yolov8n.pt',
    "YOLOv8 Trained (.pt)": r'D:\STUDY_HARD\Uni\2024.2\THESIS\Rice-Plant-Disease-Detection\runs\train\100e_20_3\weights\100e203.pt',
    "YOLOv8 ONNX (.onnx)": r'D:\STUDY_HARD\Uni\2024.2\THESIS\Rice-Plant-Disease-Detection\runs\train\100e_20_3\weights\100e203.onnx',
    "YOLOv8 Pruned (.onnx)": r'D:\STUDY_HARD\Uni\2024.2\THESIS\Rice-Plant-Disease-Detection\100e203_pruned.pt',
    "YOLOv8 Quantized (.onnx)":r'D:\STUDY_HARD\Uni\2024.2\THESIS\Rice-Plant-Disease-Detection\100e203_quantitized.onnx',
    "YOLOv8 Float16 (.onnx)":r'D:\STUDY_HARD\Uni\2024.2\THESIS\Rice-Plant-Disease-Detection\100e203_fp16.onnx',
    "YOLOv8 ghost":r'D:\STUDY_HARD\Uni\2024.2\THESIS\Rice-Plant-Disease-Detection\runs\train\ghost_100_293\weights\ghost100.pt',
    "Ghost onnx": r'D:\STUDY_HARD\Uni\2024.2\THESIS\Rice-Plant-Disease-Detection\ghost_int8_dyna.onnx',
    "ghost_fp16": r'D:\STUDY_HARD\Uni\2024.2\THESIS\Rice-Plant-Disease-Detection\ghost_fp16.onnx',
    "gema": r'D:\STUDY_HARD\Uni\2024.2\THESIS\Rice-Plant-Disease-Detection\runs\train\ema_org\weights\gema.pt',
    # Thêm các file khác nếu cần
}

def check_file_size(path):
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        return f"{size_mb:.2f} MB"
    else:
        return "❌ Không tìm thấy file"

print("📊 Kích thước các model YOLOv8:")
print("-" * 45)
for name, path in model_files.items():
    size = check_file_size(path)
    print(f"{name:<30}: {size}")
print("-" * 45)

