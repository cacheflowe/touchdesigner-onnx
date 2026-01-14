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
		model_dir = os.path.join(project.folder, 'data', 'ml', 'models', 'fastsam-x')
		model_path = os.path.join(model_dir, 'model.onnx')
		printONNX("model:", model_path)

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
	
	# COCO class names (FastSAM uses COCO dataset classes)
	COCO_CLASSES = {
		0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 
		5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
		10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird',
		15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
		20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
		25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
		30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
		35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
		40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
		45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
		50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
		55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
		60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse',
		65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven',
		70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock',
		75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
	}
	
	# Class-specific colors (RGB normalized 0-1)
	CLASS_COLORS = {
		'person': (0.0, 1.0, 0.0),      # Green for people
		'car': (1.0, 0.0, 0.0),          # Red for cars
		'bicycle': (0.0, 0.0, 1.0),      # Blue for bicycles
		'dog': (1.0, 1.0, 0.0),          # Yellow for dogs
		'cat': (1.0, 0.5, 0.0),          # Orange for cats
		'chair': (0.5, 0.0, 1.0),        # Purple for chairs
		'bottle': (0.0, 1.0, 1.0),       # Cyan for bottles
		'default': (1.0, 1.0, 1.0)       # White for other objects
	}
	
	# Classes to detect (empty list = all classes)
	CLASSES_TO_DETECT = ['person']  # Only detect people; add more like: ['person', 'car', 'dog']
	
	try:
		# Preprocessing for FastSAM (YOLO-based model)
		nA = input_tensor_cache  # This is the raw numpy array from texture
		nA = npu.flip_v(nA)
		nA = npu.rgba_to_rgb(nA)
		nA = npu.grayscale_to_rgb(nA)
		
		# Store original size for later
		original_h, original_w = nA.shape[:2]

		# Resize to model input size (640x640 for FastSAM - required by model)
		input_size = 640
		if nA.shape[:2] != (input_size, input_size):
			nA = cv2.resize(nA, (input_size, input_size), interpolation=cv2.INTER_NEAREST)  # NEAREST is faster
			
		# YOLO expects [batch, channels, height, width] format
		input_tensor = nA.transpose(2, 0, 1)  # HWC to CHW
		input_tensor = np.expand_dims(input_tensor, axis=0).astype('float32')  # Add batch
		
		# Run inference
		# FastSAM outputs may include: boxes, scores, classes, mask_coefficients, mask_prototypes
		outputs = session.run(None, {session.get_inputs()[0].name: input_tensor})
		
		# Parse outputs - structure depends on model variant
		if len(outputs) >= 4:
			boxes = outputs[0][0]  # (N, 4) or (8400, 4)
			scores = outputs[1][0]  # (N,) or (8400,)
			
			# Check if we have class predictions
			if len(outputs) >= 5:
				classes = outputs[2][0]  # (N,) or (8400,)
				mask_coeffs = outputs[3][0]  # (N, 32) or (8400, 32)
				mask_protos = outputs[4][0]  # (32, 160, 160)
			else:
				classes = None
				mask_coeffs = outputs[2][0]  # (8400, 32)
				mask_protos = outputs[3][0]  # (32, 160, 160)
		else:
			printONNX("Unexpected number of outputs:", len(outputs))
			output_img = np.zeros((original_h, original_w, 3), dtype=np.float32)
			output_img = npu.flip_v(output_img)
			with inference_lock:
				pending_result = output_img
			return
		
		# Filter by confidence threshold - higher = faster (fewer masks to process)
		conf_threshold = 0.85
		valid_indices = scores > conf_threshold
		
		# Additional class filtering if classes are available
		if classes is not None and len(CLASSES_TO_DETECT) > 0:
			class_filter = np.zeros_like(valid_indices, dtype=bool)
			for class_idx, class_name in COCO_CLASSES.items():
				if class_name in CLASSES_TO_DETECT:
					class_filter |= (classes == class_idx)
			valid_indices &= class_filter
		
		if np.any(valid_indices):
			# Get valid detections
			valid_coeffs = mask_coeffs[valid_indices]  # (N, 32)
			valid_classes = classes[valid_indices] if classes is not None else None
			valid_scores = scores[valid_indices]
			
			# Decode masks: multiply coefficients with prototypes
			# Result shape: (N, 160, 160)
			masks = np.matmul(valid_coeffs, mask_protos.reshape(32, -1)).reshape(-1, 160, 160)
			
			# Apply sigmoid to get probabilities
			masks = 1 / (1 + np.exp(-masks))
			
			# Threshold masks to get binary masks (only keep high-confidence regions)
			mask_threshold = 0.25
			binary_masks = (masks > mask_threshold).astype(np.float32)
			
			# Filter by mask area - only keep larger segments (likely to be people/main objects)
			min_area_ratio = 0.03  # Minimum 3% of image area - higher = faster
			max_masks = 5  # Limit to top 5 largest masks for speed
			
			# Vectorized area calculation - much faster than list comprehension
			mask_areas = binary_masks.sum(axis=(1, 2))
			min_area = (160 * 160) * min_area_ratio
			
			# Filter by area
			area_filter = mask_areas > min_area
			filtered_masks = binary_masks[area_filter]
			filtered_scores = valid_scores[area_filter]
			filtered_classes = valid_classes[area_filter] if valid_classes is not None else None
			filtered_areas = mask_areas[area_filter]
			
			# Sort by area (largest first) and take top N
			if len(filtered_masks) > max_masks:
				top_indices = np.argsort(filtered_areas)[-max_masks:]
				filtered_masks = filtered_masks[top_indices]
				filtered_scores = filtered_scores[top_indices]
				filtered_classes = filtered_classes[top_indices] if filtered_classes is not None else None
				filtered_areas = filtered_areas[top_indices]
			
			printONNX(f"Found {len(filtered_masks)} objects after filtering (from {np.sum(valid_indices)} detections)")
			printONNX(f"Score range: {filtered_scores.min():.3f} to {filtered_scores.max():.3f}")
			if filtered_classes is not None:
				printONNX(f"Classes: {[COCO_CLASSES.get(int(c), 'unknown') for c in filtered_classes[:5]]}")
			
			binary_masks = filtered_masks
			valid_classes = filtered_classes
			
			# Create a color palette for different segments
			# Using distinct colors for better visualization
			SEGMENT_COLORS = [
				(1.0, 0.0, 0.0),    # Red
				(0.0, 1.0, 0.0),    # Green
				(0.0, 0.0, 1.0),    # Blue
				(1.0, 1.0, 0.0),    # Yellow
				(1.0, 0.0, 1.0),    # Magenta
				(0.0, 1.0, 1.0),    # Cyan
				(1.0, 0.5, 0.0),    # Orange
				(0.5, 0.0, 1.0),    # Purple
				(0.0, 1.0, 0.5),    # Spring Green
				(1.0, 0.0, 0.5),    # Rose
				(0.5, 1.0, 0.0),    # Lime
				(0.0, 0.5, 1.0),    # Sky Blue
			]
			
			# Create colored output - start with black background
			output_img = np.zeros((160, 160, 3), dtype=np.float32)
			
			# Vectorized color application - much faster than looping
			for i, mask in enumerate(binary_masks):
				# Cycle through color palette
				color = SEGMENT_COLORS[i % len(SEGMENT_COLORS)]
				# Apply color directly using broadcasting (faster than stack+multiply)
				for c in range(3):
					output_img[:, :, c] = np.maximum(output_img[:, :, c], mask * color[c])
			
			printONNX(f"Output range: {output_img.min():.3f} to {output_img.max():.3f}")
			
			# Resize to original dimensions using faster interpolation
			output_img = cv2.resize(output_img, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
		else:
			# No detections above threshold
			output_img = np.zeros((original_h, original_w, 3), dtype=np.float32)
		
		output_img = npu.flip_v(output_img)
		
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
			# printONNX("Inference thread completed.", frames_skipped_final, "frames were skipped.")
			op('constant_performance').par.const0value = frames_skipped_final

			# Ensure output is float32 for TouchDesigner
			output_img = output_img.astype(np.float32)
			
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
