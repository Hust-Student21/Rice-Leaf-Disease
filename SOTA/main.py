import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
 
if __name__ == '__main__':
 
    model = YOLO(r'D:\STUDY_HARD\Uni\2024.2\THESIS\Rice-Plant-Disease-Detection\QUANTIZE\custom_yolov8_int8_static.onnx')
 
    # model.load('yolov8n.pt') # 是否加载预训练权重，科研不建加载否则很难提升精度
 
    model.predict(
        data=r"C:\Users\ADMIN\OneDrive - Hanoi University of Science and Technology\Desktop\Lab - Copy\rice_disease\rice_dataset.yaml",
        cache=False,
        imgsz=640,
        epochs=1,
        single_cls=False,  # 是否是单类别检测
        batch=8,
        close_mosaic=0,
        workers=0,
        device='0',
        optimizer='Adam',   # using SGD
        # resume='runs/train/exp/weights/last.pt',   # 如过想续训就设置 last.pt 的地址
        amp=False,                                   # 如果出现训练损失为 Nan 可以关闭 amp
        project='runs/detect',
        name='test_quantize',
        
    )
    