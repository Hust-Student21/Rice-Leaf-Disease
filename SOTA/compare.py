
import os

# Danh sách các file model muốn kiểm tra
model_files = {
    "gema": r'D:\STUDY_HARD\Uni\2024.2\THESIS\Rice-Plant-Disease-Detection\runs\train\ema_org\weights\gema.pt',
    "gema_again": r'D:\STUDY_HARD\Uni\2024.2\THESIS\Rice-Plant-Disease-Detection\runs\train\ghost_ema_3disease_org_retrain\weights\gema.pt',
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




