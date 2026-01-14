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

# Model Configuration -------------------------------
# Switch between 'general' (256x256) and 'landscape' (256x144) models
MODEL_TYPE = 'landscape'  # Options: 'general' or 'landscape'
ENABLE_POST_PROCESSING = False  # Enable erosion + bilateral filter for cleaner masks

MODEL_CONFIGS = {
	'general': {
		'filename': 'model.onnx',
		'input_size': (256, 256),  # (width, height)
		'format': 'NCHW',  # batch, channels, height, width
	},
	'landscape': {
		'filename': 'selfie_segmentation_landscape.onnx',
		'input_size': (256, 144),  # (width, height)
		'format': 'NHWC',  # batch, height, width, channels
	}
}

# Get active config
config = MODEL_CONFIGS[MODEL_TYPE]

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
		printONNX(f"Starting ONNX model loading ({MODEL_TYPE} model)...")

		# Build paths using MODEL_TYPE config
		model_dir = os.path.join(project.folder, 'data', 'ml', 'models', 'mediapipe-selfie-segmentation')
		model_path = os.path.join(model_dir, config['filename'])
		printONNX("model:", model_path)
		printONNX(f"Config: {config['input_size'][0]}x{config['input_size'][1]}, format: {config['format']}")

		# load model & provider
		# SessionOptions for model configuration
		sess_options = ort.SessionOptions()
		
		onnx_util.log_onnx_options()
		providers = onnx_util.providers()
		# ONNX Runtime automatically looks for external data files (like model.data) 
		# in the same directory as the model file
		temp_session = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
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
		# Preprocessing for MediaPipe Selfie Segmentation
		nA = input_tensor_cache
		
		# Flip and convert color
		# nA = npu.flip_v(nA)
		nA = npu.rgba_to_rgb(nA)
		nA = npu.grayscale_to_rgb(nA)
		
		# MediaPipe models expect BGR format - now doing this in TouchDesigner input
		# nA = cv2.cvtColor(nA, cv2.COLOR_RGB2BGR)
		
		# Store original size
		original_h, original_w = nA.shape[:2]

		# Resize to model input size (width, height from config)
		input_size = config['input_size']
		if nA.shape[:2] != (input_size[1], input_size[0]):  # shape is (height, width)
			printONNX(f"Resizing input from {nA.shape[1]}x{nA.shape[0]} to {input_size[0]}x{input_size[1]}")
			nA = cv2.resize(nA, input_size, interpolation=cv2.INTER_NEAREST)  # cv2.resize takes (width, height)
		
		# Input is already normalized to 0-1 from TouchDesigner
		input_tensor = nA.astype('float32')
		
		# Format conversion based on model type
		if config['format'] == 'NCHW':
			# Model expects [batch, channels, height, width] format
			input_tensor = input_tensor.transpose(2, 0, 1)  # HWC to CHW
			input_tensor = np.expand_dims(input_tensor, axis=0)  # Add batch dimension
		else:  # NHWC
			# Model expects [batch, height, width, channels] format
			input_tensor = np.expand_dims(input_tensor, axis=0)  # Add batch dimension
		
		# Run inference
		# MediaPipe Selfie Segmentation outputs a single channel segmentation mask
		outputs = session.run(None, {session.get_inputs()[0].name: input_tensor})
		
		# Debug output shape
		# printONNX(f"Output shape: {outputs[0].shape}, min: {outputs[0].min():.4f}, max: {outputs[0].max():.4f}")
		
		# Get the segmentation mask
		mask = outputs[0]
		
		# Remove batch dimension if present
		if mask.shape[0] == 1:
			mask = mask[0]
		
		# printONNX(f"After batch removal: {mask.shape}")
		
		# Handle different output formats
		if len(mask.shape) == 3:
			# Check if last dimension is channels (should be 1 or 2)
			if mask.shape[2] <= 2:
				# Format is (H, W, C) - channel last
				mask = mask[:, :, 0]
			elif mask.shape[0] <= 2:
				# Format is (C, H, W) - channel first
				mask = mask[0, :, :]
			else:
				# Ambiguous - assume channel last if last dim is smallest
				if mask.shape[2] < mask.shape[0] and mask.shape[2] < mask.shape[1]:
					mask = mask[:, :, 0]
				else:
					mask = mask[0, :, :]
		
		# printONNX(f"Final mask shape: {mask.shape}, min: {mask.min():.4f}, max: {mask.max():.4f}, mean: {mask.mean():.4f}")
		
		# Fast threshold - no blur, no morphology for speed
		threshold = 0.1
		final_mask = (mask > threshold).astype(np.float32)
		
		# Optional post-processing pipeline (based on SelfieBarracuda implementation)
		if ENABLE_POST_PROCESSING:
			# 1. Erosion - removes small noise/specks
			kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
			final_mask = cv2.erode(final_mask, kernel, iterations=1)
			
			# 2. Bilateral filter - edge-preserving smooth (preserves boundaries)
			# Params: diameter, sigmaColor, sigmaSpace
			final_mask = cv2.bilateralFilter(final_mask, 5, 0.1, 5)
			
			# Optional: Re-threshold after bilateral to get back to binary if needed
			# final_mask = (final_mask > 0.5).astype(np.float32)
		
		# Convert to RGB output
		output_img = np.stack([final_mask * 0.0, final_mask * 1.0, final_mask * 0.0], axis=2)  # Green
		# Or grayscale: output_img = np.stack([final_mask, final_mask, final_mask], axis=2)
		
		# Resize to original dimensions using fast interpolation
		# output_img = cv2.resize(output_img, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
		
		# Flip output back to match TouchDesigner orientation
		# output_img = npu.flip_v(output_img)
		
		# Store results thread-safely
		with inference_lock:
			pending_result = output_img
			
	except Exception as e:
		printONNX(f"Inference error: {e}")
		import traceback
		printONNX(traceback.format_exc())
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
			op('constant_performance').par.const0value = frames_skipped_final
			if frames_skipped_final > 0: # prevent div by zero
				op('constant_performance').par.const1value = math.floor(60 / frames_skipped_final)

			# Ensure output is float32 for TouchDesigner
			output_img = output_img.astype(np.float32)
			
			# Output result directly (already fully processed)
			scriptOp.copyNumpyArray(output_img)
			return  # Early return after outputting result

	# If inference is still running, skip this frame (natural frame skipping via threading)
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
	
	# Debug: Check thread count
	active_threads = threading.active_count()
	if active_threads > 10:
		printONNX(f"Warning: {active_threads} active threads!")
