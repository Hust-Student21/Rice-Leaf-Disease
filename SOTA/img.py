from ultralytics import YOLO

# Load a model
model = YOLO(r"D:\STUDY_HARD\Uni\2024.2\THESIS\Rice-Plant-Disease-Detection\LSKA\LSKA_ghost_SGD_28.5\weights\lska.pt")  # pretrained YOLO11n model

model.model

print(model.model.model[0])  # print model names
