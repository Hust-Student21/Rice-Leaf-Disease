from ultralytics import YOLO

model = YOLO(r'D:\STUDY_HARD\Uni\2024.2\THESIS\Rice-Plant-Disease-Detection\runs\train\ema_org\weights\gema.pt')  # or your custom-trained model

 # Export to ONNX
model.export(format='onnx', opset=12, dynamic=True)

#from onnxruntime.quantization import quantize_dynamic, QuantType

#quantize_dynamic(
 #   model_input=r'D:\STUDY_HARD\Uni\2024.2\THESIS\Rice-Plant-Disease-Detection\runs\train\ghost_100_293\weights\ghost100.onnx',
  #  model_output="ghost_int8.onnx",
   # weight_type=QuantType.QInt8
#)

