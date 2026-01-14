import os
import sys

# import other dependencies now that the path supports it
import threading
import numpy as np
import onnxruntime as ort
import cv2

# custom util imports
onnx_util = mod(f'{op.PyUtils}/onnx_util')
npu = mod(f'{op.PyUtils}/numpy_util')

# Threaded model-loading helpers -------------------------------

loading_thread = None
is_loading = False
load_error = None

# Threaded inference state -------------------------------

inference_thread = None
is_inferencing = False
inference_lock = threading.Lock()
pending_result = None  # Results from background thread
input_tensor_cache = None  # Pre-processed input for thread
frames_skipped = 0  # Track how many frames we've skipped
frames_skipped_final = 0 # Final count of skipped frames to report on UI thread

# ONNX setup -------------------------------

ort.preload_dlls(directory="")
session = None  # ONNX session

def printONNX(*args):
	print("[ONNX]", *args)

def loadONNX(scriptOp):
	global session, loading_thread, is_loading

	if is_loading:
		printONNX("Model is already loading...") 
		return

	# Reset session and start loading thread
	session = None
	scriptOp.par.Loadstatus = "loading"
	loading_thread = threading.Thread(target=_load_model_thread)
	loading_thread.daemon = True
	loading_thread.start()


def _load_model_thread():
	global session, is_loading, load_error

	is_loading = True
	load_error = None

	try:
		printONNX('=============================================')
		printONNX("Starting ONNX model loading in background...")

		# Build paths & config
		model_path = os.path.join(project.folder, 'data', 'ml', 'models', 'resnet50-human', 'deeplabv3p-resnet50-human.onnx')
		printONNX("model:", model_path)

		# load model & provider
		onnx_util.log_onnx_options()
		providers = onnx_util.providers()
		temp_session = ort.InferenceSession(model_path, providers=providers)
		printONNX('ONNX Device activated:', ort.get_device())
		printONNX('### session props -----------------------------------')
		onnx_util.log_model_details(temp_session)
		# Only assign to global session when fully loaded
		session = temp_session
		printONNX("ONNX model loaded successfully!")
		printONNX('=============================================')

	except Exception as e:
		load_error = str(e)
		printONNX(f"Error loading ONNX model: {e}")
	finally:
		is_loading = False


def get_loading_status():
	"""Returns status of model loading"""
	if session is not None:
		return "loaded"
	elif is_loading:
		return "loading"
	elif load_error:
		return f"error: {load_error}"
	else:
		return "not_loaded"


# Shared Script Op callbacks -------------------------------

# press 'Setup Parameters' in the OP to call this function to re-create the parameters.
def onSetupParameters(scriptOp):
	page = scriptOp.appendCustomPage('Custom')
	# add reload pulse
	page.appendPulse('Reloadonnx', label='Reload ONNX')
	# add status info
	page.appendStr('Loadstatus', label='Load Status')
	scriptOp.par.Loadstatus = get_loading_status()
	return


# called whenever custom pulse parameter is pushed
def onPulse(par):
	if par.name == 'Reloadonnx':
		session = None  # reset the session
	return


def _inference_thread():
	"""Background thread for preprocessing, ONNX inference, and post-processing."""
	global pending_result, is_inferencing, frames_skipped_final, frames_skipped
	
	try:
		# Preprocessing (all numpy operations)
		nA = input_tensor_cache  # This is the raw numpy array from texture
		nA = npu.flip_v(nA)
		nA = npu.rgba_to_rgb(nA)
		nA = npu.grayscale_to_rgb(nA)
		
		# TouchDesigner textures are already 0-1 float, so denormalize to 0-255
		nA = npu.denormalize_td_image(nA)

		# Resize to model input size (512x512 for this model)
		# if nA.shape[:2] != (512, 512):
		# 	nA = cv2.resize(nA, (512, 512), interpolation=cv2.INTER_CUBIC)
		
		# PyTorch standardization: (x / 255 - mean) / std
		nA = nA.astype(np.float32)
		mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
		std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
		nA = (nA / 255.0 - mean) / std
		
		# This model expects [batch, height, width, channels] format (TensorFlow-style)
		# Just add batch dimension, keep HWC format
		input_tensor = np.expand_dims(nA, axis=0).astype('float32')
		
		# Run inference
		input_name = session.get_inputs()[0].name
		output_name = session.get_outputs()[0].name
		output = session.run([output_name], {input_name: input_tensor})[0]
		
		# Post-processing for segmentation mask
		# TensorFlow DeepLabV3 outputs [batch, height, width, classes]
		# Remove batch dimension
		if output.shape[0] == 1:
			output = output[0]  # Now [height, width, classes]
		
		# Get the predicted class for each pixel (argmax along last dimension)
		if len(output.shape) == 3:  # [height, width, classes]
			predicted_classes = np.argmax(output, axis=-1)  # [height, width]
		else:  # Already a single channel
			predicted_classes = output
		
		# Create color map for different body parts
		# 0: background (black), 1: unknown, 2: hair, 3: unknown, 4: glasses,
		# 5: top-clothes, 6-8: unknown, 9: bottom-clothes, 10: torso-skin,
		# 11-12: unknown, 13: face, 14-15: arms, 16-17: legs, 18-19: feet
		color_map = np.array([
			[0.0, 0.0, 0.0],      # 0: background - black
			[0.5, 0.5, 0.5],      # 1: unknown - gray
			[0.8, 0.3, 0.3],      # 2: hair - red
			[0.5, 0.5, 0.5],      # 3: unknown - gray
			[0.3, 0.8, 0.8],      # 4: glasses - cyan
			[0.3, 0.3, 0.8],      # 5: top-clothes - blue
			[0.5, 0.5, 0.5],      # 6: unknown - gray
			[0.5, 0.5, 0.5],      # 7: unknown - gray
			[0.5, 0.5, 0.5],      # 8: unknown - gray
			[0.8, 0.3, 0.8],      # 9: bottom-clothes - magenta
			[0.9, 0.7, 0.6],      # 10: torso-skin - skin tone
			[0.5, 0.5, 0.5],      # 11: unknown - gray
			[0.5, 0.5, 0.5],      # 12: unknown - gray
			[1.0, 0.8, 0.7],      # 13: face - light skin tone
			[0.3, 0.8, 0.3],      # 14: left-arm - green
			[0.3, 0.8, 0.3],      # 15: right-arm - green
			[0.8, 0.8, 0.3],      # 16: left-leg - yellow
			[0.8, 0.8, 0.3],      # 17: right-leg - yellow
			[0.8, 0.5, 0.3],      # 18: left-foot - orange
			[0.8, 0.5, 0.3],      # 19: right-foot - orange
		], dtype=np.float32)
		
		# Map predicted classes to colors
		output_img = color_map[predicted_classes]
		
		# Already 0-1 float for TouchDesigner
		output_img = npu.flip_v(output_img)
		
		# Store results thread-safely
		with inference_lock:
			pending_result = output_img
			
	except Exception as e:
		print(f"Inference error: {e}")
	finally:
		is_inferencing = False
		frames_skipped_final = frames_skipped


def onCook(scriptOp):
	global session, is_loading, load_error, is_inferencing, pending_result, input_tensor_cache, frames_skipped, frames_skipped_final

	# Update status parameter
	status = get_loading_status()
	scriptOp.par.Loadstatus = status

	# make sure we've loaded the model
	if session is None:
		if not is_loading:
			loadONNX(scriptOp)
		# Return early if model isn't ready yet
		return

	# Check if we have a loading error
	if load_error:
		printONNX(f"Cannot process: {load_error}")
		return

	# Check if we have results from background thread
	with inference_lock:
		if pending_result is not None:
			output_img = pending_result
			pending_result = None
			frames_skipped = 0
			# printONNX("Inference thread completed.", frames_skipped_final, "frames were skipped.")
			op('constant_performance').par.const0value = frames_skipped_final

			# Output result directly (already fully processed)
			scriptOp.copyNumpyArray(output_img)

	# If inference is still running, skip this frame
	if is_inferencing:
		frames_skipped += 1
		return

	# Capture input on main thread (GPU texture access only)
	try:
		inputTex = scriptOp.inputs[0]
		nA = inputTex.numpyArray(delayed=True)
		if nA is None:
			return
		
		# Store raw array for background thread to process
		input_tensor_cache = nA

	except Exception as e:
		printONNX(f"Error capturing input: {e}")
		return

	# Start inference in background thread
	is_inferencing = True
	inference_thread = threading.Thread(target=_inference_thread)
	inference_thread.daemon = True
	inference_thread.start()
