from ultralytics import YOLO
model = YOLO(r'D:\STUDY_HARD\Uni\2024.2\THESIS\Rice-Plant-Disease-Detection\runs\train\100e_20_3\weights\100e203.pt')  # load an official model

   # Export the model
model.export(format="tfjs")            
