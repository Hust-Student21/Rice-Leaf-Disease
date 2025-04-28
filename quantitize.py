import onnx

model_fp16_path = r'D:\STUDY_HARD\Uni\2024.2\THESIS\Rice-Plant-Disease-Detection\runs\train\ghost_100_293\weights\ghost100.onnx'
model = onnx.load(model_fp16_path)

conv_nodes = []
for node in model.graph.node:
    if node.op_type == "Conv":
        conv_nodes.append(node.name)
conv_nodes

# ['/model/conv1/Conv',
#  '/model/layer1/layer1.0/conv1/Conv',
#  '/model/layer1/layer1.0/conv2/Conv',
#  '/model/layer1/layer1.1/conv1/Conv',
#  '/model/layer1/layer1.1/conv2/Conv',
#  '/model/layer2/layer2.0/conv1/Conv',
#  '/model/layer2/layer2.0/conv2/Conv',
#  '/model/layer2/layer2.0/downsample/downsample.0/Conv',
#  '/model/layer2/layer2.1/conv1/Conv',
#  '/model/layer2/layer2.1/conv2/Conv',
#  '/model/layer3/layer3.0/conv1/Conv',
#  '/model/layer3/layer3.0/conv2/Conv',
#  '/model/layer3/layer3.0/downsample/downsample.0/Conv',
#  '/model/layer3/layer3.1/conv1/Conv',
#  '/model/layer3/layer3.1/conv2/Conv',
#  '/model/layer4/layer4.0/conv1/Conv',
#  '/model/layer4/layer4.0/conv2/Conv',
#  '/model/layer4/layer4.0/downsample/downsample.0/Conv',
#  '/model/layer4/layer4.1/conv1/Conv',
#  '/model/layer4/layer4.1/conv2/Conv',
#  '/pool/Conv']

from onnxruntime.quantization import quantize_dynamic, QuantType

model_int8_path = "ghost_int8_dyna.onnx"
quantized_model = quantize_dynamic(
    model_fp16_path,  # Input model
    model_int8_path,  # Output model
    weight_type=QuantType.QInt8,
    nodes_to_exclude=conv_nodes,
)
print("Dynamic Quantization Complete!")

