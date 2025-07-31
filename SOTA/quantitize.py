from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input=r"D:\STUDY_HARD\Uni\2024.2\THESIS\Rice-Plant-Disease-Detection\LSKA\Thesis.onnx",
    model_output=r"TThesis_int8.onnx",
    weight_type=QuantType.QInt8
)
