from ultralytics import YOLO

# Load a YOLO11n PyTorch model
model = YOLO(r"D:\STUDY_HARD\Uni\2024.2\THESIS\Rice-Plant-Disease-Detection\runs\train\100e_20_3\weights\100e203.pt")

# Export the model to NCNN format
model.export(format="ncnn")  # creates 'yolo11n_ncnn_model'

# Load the exported NCNN model
ncnn_model = YOLO("100e203_ncnn_model")

# Run inference
#results = ncnn_model(r"D:\STUDY_HARD\Uni\2024.2\THESIS\Rice-Plant-Disease-Detection\blast.jpg")
#taskresults.show()  # display inference results