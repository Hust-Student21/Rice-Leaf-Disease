import numpy as np
import onnxruntime as ort
import cv2
import torch
import torchvision
import yaml
from pathlib import Path
import time

# ========== CONFIG ==========
MODEL_PATH = r"D:\STUDY_HARD\Uni\2024.2\THESIS\Rice-Plant-Disease-Detection\QUANTIZE\custom_yolov8_int8_static.onnx"
IMAGE_PATH = r"D:\STUDY_HARD\Uni\2024.2\THESIS\Rice-Plant-Disease-Detection\blast.jpg"
YAML_PATH = r"C:\Users\ADMIN\OneDrive - Hanoi University of Science and Technology\Desktop\Lab - Copy\rice_disease\rice_dataset.yaml"
IMG_SIZE = 640
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
# ============================

def load_class_names(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    names = data.get('names', {})
    if isinstance(names, dict):
        return [names[i] for i in sorted(names)]
    return names

CLASS_NAMES = load_class_names(YAML_PATH)

def preprocess(img_path, img_size=640):
    img = cv2.imread(img_path)
    orig = img.copy()
    img = cv2.resize(img, (img_size, img_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)  # Add batch dim
    return img, orig

def xywh2xyxy(x):
    y = x.clone()
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def non_max_suppression_custom(preds, conf_thres=0.25, iou_thres=0.45):
    pred = torch.tensor(preds[0]).squeeze(0).transpose(0, 1)  # [8, 8400] → [8400, 8]
    boxes = pred[:, 0:4]
    objectness = pred[:, 4]
    class_scores = pred[:, 5:]  # shape: [8400, 3]

    scores, class_ids = class_scores.max(1)
    conf = objectness * scores

    mask = conf > conf_thres
    boxes = boxes[mask]
    conf = conf[mask]
    class_ids = class_ids[mask]

    if boxes.shape[0] == 0:
        return []

    boxes = xywh2xyxy(boxes)
    detections = torch.cat((boxes, conf.unsqueeze(1), class_ids.unsqueeze(1).float()), 1)
    keep = torchvision.ops.nms(detections[:, :4], detections[:, 4], iou_thres)
    return detections[keep].cpu().numpy()


def postprocess(preds, orig_img, scale):
    dets = non_max_suppression_custom(preds, CONF_THRESHOLD, IOU_THRESHOLD)
    h, w = orig_img.shape[:2]

    for x1, y1, x2, y2, conf, cls in dets:
        if int(cls) >= len(CLASS_NAMES): continue
        label = f"{CLASS_NAMES[int(cls)]}: {conf:.2f}"
        x1 = int(x1 / scale)
        x2 = int(x2 / scale)
        y1 = int(y1 / scale)
        y2 = int(y2 / scale)
        cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(orig_img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    return orig_img

    return orig_img

def run():
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(MODEL_PATH, providers=providers)
    input_name = session.get_inputs()[0].name

    img_tensor, orig_img = preprocess(IMAGE_PATH, IMG_SIZE)
    scale = orig_img.shape[1] / IMG_SIZE

    start = time.time()
    preds = session.run(None, {input_name: img_tensor})
    for i, p in enumerate(preds):
        print(f"Output[{i}] shape: {p.shape}")
    end = time.time()
    print(f"Inference time: {(end - start)*1000:.2f} ms")

    out_img = postprocess(preds, orig_img, scale)
    cv2.imwrite("result.jpg", out_img)
    print("✅ Result saved as result.jpg")

if __name__ == "__main__":
    run()
