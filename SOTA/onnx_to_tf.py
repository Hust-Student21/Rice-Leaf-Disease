from onnx_tf.backend import prepare
import onnx

# Load your ONNX model
onnx_model = onnx.load(r'D:\STUDY_HARD\Uni\2024.2\THESIS\Rice-Plant-Disease-Detection\NORMAL\100e203.onnx')  # use your actual ONNX filename
tf_rep = prepare(onnx_model)

# Export to TensorFlow SavedModel format
tf_rep.export_graph("Normal_tf_model")
