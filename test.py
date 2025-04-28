from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO(r"D:\STUDY_HARD\Uni\2024.2\THESIS\Rice-Plant-Disease-Detection\runs\train\ghost_ema_3disease_org_retrain\weights\best.pt")

# Define path to the image file
source = r"D:\STUDY_HARD\Uni\2024.2\THESIS\Rice-Plant-Disease-Detection\blast.jpg"

# Run inference on the source
results = model(source)  # list of Results objects
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="test.jpg")  # save to disk