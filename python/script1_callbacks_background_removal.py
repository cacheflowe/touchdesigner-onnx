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
		model_path = os.path.join(project.folder, 'data', 'ml', 'models', 'person-u2netp', 'person-u2netp.onnx')
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

		# Resize to model input size
		if nA.shape[:2] != (384, 384):
			nA = cv2.resize(nA, (384, 384), interpolation=cv2.INTER_CUBIC)

		# Convert BGR to RGB (cv2 operations expect BGR order)
		# nA = cv2.cvtColor(nA, cv2.COLOR?_BGR2RGB)
		
		# PyTorch standardization: (x / 255 - mean) / std
		nA = nA.astype(np.float32)
		mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
		std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
		nA = (nA / 255.0 - mean) / std
		
		# Transpose to CHW format and add batch dimension
		input_tensor = nA.transpose(2, 0, 1)
		input_tensor = input_tensor.reshape(-1, 3, 384, 384).astype('float32')
		
		# Run inference
		input_name = session.get_inputs()[0].name
		output_name = session.get_outputs()[0].name
		depth_map = session.run([output_name], {input_name: input_tensor})[0]
		
		# Post-processing (squeeze and normalize)
		output_img = np.squeeze(depth_map)
		
		# Min-max normalization to 0-255 range
		min_value = np.min(output_img)
		max_value = np.max(output_img)
		output_img = (output_img - min_value) / (max_value - min_value)
		output_img *= 255.0
		output_img = output_img.astype(np.uint8)
		
		# Convert grayscale to RGB (stack 3 channels)
		output_img = np.stack([output_img, output_img, output_img], axis=-1)
		
		# Convert back to 0-1 float for TouchDesigner
		output_img = output_img.astype(np.float32) / 255.0
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
