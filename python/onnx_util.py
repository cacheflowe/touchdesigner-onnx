import numpy
import cv2
import onnxruntime as ort


def printONNX(*args):
    print("[ONNX]", *args)

def log_onnx_options():
	printONNX('version', ort.__version__)
	# print('Available providers:')
	# for provider in ort.get_available_providers():
	# 	print('-', provider)

def providers():
    return ['CUDAExecutionProvider', 'CPUExecutionProvider'] # 'TensorrtExecutionProvider'

def log_model_details(session):
    printONNX('Session providers:', session.get_providers())
    # printONNX('- Model description:', session.get_modelmeta().description)
    # printONNX('- Model version:', session.get_modelmeta().version)
    printONNX('Inputs: -----------------')
    for i in session.get_inputs():
        printONNX('-', i.name, i.shape, i.type)
    printONNX("Outputs: ----------------")
    for o in session.get_outputs():
        printONNX('-', o.name, o.shape, o.type)
    printONNX("Input shape: ------------")
    input_shape = session.get_inputs()[0].shape
    has_dynamic_dims = any(isinstance(dim, str) for dim in input_shape)
    if has_dynamic_dims:
        # Use default size for MoveNet multipose (256x256)
        height, width = 256, 256
        printONNX(f"Model input shape is dynamic! Using default size: {height}x{width}")
    else:
        # Use the model's expected dimensions if they're specified
        batch_size, channels, height, width = input_shape
        printONNX(f"Model expects input shape: {input_shape}")

