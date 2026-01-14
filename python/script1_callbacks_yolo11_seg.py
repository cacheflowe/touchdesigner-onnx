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
		model_dir = os.path.join(project.folder, 'data', 'ml', 'models', 'yolo11-seg')
		model_path = os.path.join(model_dir, 'yolo11s-seg.onnx')
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
		# Preprocessing for YOLO11-seg model
		nA = input_tensor_cache  # This is the raw numpy array from texture
		nA = npu.flip_v(nA)
		nA = npu.rgba_to_rgb(nA)
		nA = npu.grayscale_to_rgb(nA)
		
		# Store original size for later
		original_h, original_w = nA.shape[:2]

		# Resize to model input size (640x640 for YOLO11)
		input_size = 640
		if nA.shape[:2] != (input_size, input_size):
			nA = cv2.resize(nA, (input_size, input_size), interpolation=cv2.INTER_NEAREST)
			
		# YOLO expects [batch, channels, height, width] format
		input_tensor = nA.transpose(2, 0, 1)  # HWC to CHW
		input_tensor = np.expand_dims(input_tensor, axis=0).astype('float32')  # Add batch
		
		# Run inference
		outputs = session.run(None, {session.get_inputs()[0].name: input_tensor})
		
		# YOLO11-seg outputs: 
		# output0: [1, 116, 8400] - combined detections (4 box coords + 80 classes + 32 mask coeffs)
		# output1: [1, 32, 160, 160] - mask prototypes
		
		if len(outputs) != 2:
			printONNX(f"Unexpected number of outputs: {len(outputs)}")
			output_img = np.zeros((original_h, original_w, 3), dtype=np.float32)
			output_img = npu.flip_v(output_img)
			with inference_lock:
				pending_result = output_img
			return
		
		# Parse YOLO11-seg output format
		predictions = outputs[0][0]  # Shape: (116, 8400)
		mask_protos = outputs[1][0]  # Shape: (32, 160, 160)
		
		# Extract components from predictions
		boxes = predictions[:4, :].T  # (8400, 4) - x, y, w, h
		class_scores = predictions[4:84, :].T  # (8400, 80) - class probabilities
		mask_coeffs = predictions[84:, :].T  # (8400, 32) - mask coefficients
		
		# Get class IDs and confidence scores
		class_ids = np.argmax(class_scores, axis=1)  # (8400,)
		confidences = np.max(class_scores, axis=1)  # (8400,)
		
		# Filter by confidence threshold - higher = faster
		conf_threshold = 0.5  # Increased for better performance
		valid_indices = confidences > conf_threshold
		
		# Additional class filtering if specified
		if len(CLASSES_TO_DETECT) > 0:
			class_filter = np.zeros_like(valid_indices, dtype=bool)
			for class_idx, class_name in COCO_CLASSES.items():
				if class_name in CLASSES_TO_DETECT:
					class_filter |= (class_ids == class_idx)
			valid_indices &= class_filter
		
		if np.any(valid_indices):
			# Get valid detections
			valid_boxes = boxes[valid_indices]
			valid_coeffs = mask_coeffs[valid_indices]
			valid_classes = class_ids[valid_indices]
			valid_scores = confidences[valid_indices]
			
			# printONNX(f"Found {len(valid_boxes)} detections")
			# printONNX(f"Classes: {[COCO_CLASSES.get(int(c), 'unknown') for c in valid_classes[:20]]}")
			
			# Decode masks: multiply coefficients with prototypes
			masks = np.matmul(valid_coeffs, mask_protos.reshape(32, -1)).reshape(-1, 160, 160)
			
			# Apply sigmoid to get probabilities
			masks = 1 / (1 + np.exp(-masks))
			
			# Threshold masks
			mask_threshold = 0.5
			binary_masks = (masks > mask_threshold).astype(np.float32)
			
			# Filter by mask area - higher values = faster
			min_area_ratio = 0.02  # Minimum 2% of image area
			max_masks = 5  # Limit to 5 for speed
			
			mask_areas = binary_masks.sum(axis=(1, 2))
			min_area = (160 * 160) * min_area_ratio
			
			area_filter = mask_areas > min_area
			filtered_masks = binary_masks[area_filter]
			filtered_scores = valid_scores[area_filter]
			filtered_classes = valid_classes[area_filter]
			filtered_areas = mask_areas[area_filter]
			
			# Sort by score (highest first) and take top N
			if len(filtered_masks) > max_masks:
				top_indices = np.argsort(filtered_scores)[-max_masks:]
				filtered_masks = filtered_masks[top_indices]
				filtered_scores = filtered_scores[top_indices]
				filtered_classes = filtered_classes[top_indices]
			
			# printONNX(f"After filtering: {len(filtered_masks)} objects")
			# printONNX(f"Final classes: {[COCO_CLASSES.get(int(c), 'unknown') for c in filtered_classes]}")
			
			# Create a color palette
			SEGMENT_COLORS = [
				(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0, 0.0),
				(1.0, 0.0, 1.0), (0.0, 1.0, 1.0), (1.0, 0.5, 0.0), (0.5, 0.0, 1.0),
				(0.0, 1.0, 0.5), (1.0, 0.0, 0.5), (0.5, 1.0, 0.0), (0.0, 0.5, 1.0),
			]
			
			# Create colored output
			output_img = np.zeros((160, 160, 3), dtype=np.float32)
			
			# Apply colors based on class
			for i, (mask, class_id) in enumerate(zip(filtered_masks, filtered_classes)):
				# Get class name and use class-specific color if available
				class_name = COCO_CLASSES.get(int(class_id), 'default')
				if class_name in CLASS_COLORS:
					color = CLASS_COLORS[class_name]
				else:
					# Use palette color for other classes
					color = SEGMENT_COLORS[i % len(SEGMENT_COLORS)]
				
				# Apply color
				for c in range(3):
					output_img[:, :, c] = np.maximum(output_img[:, :, c], mask * color[c])
			
			# Resize to original dimensions using smoother interpolation for better quality
			output_img = cv2.resize(output_img, (original_w, original_h), interpolation=cv2.INTER_CUBIC)
		else:
			# No detections
			output_img = np.zeros((original_h, original_w, 3), dtype=np.float32)
		
		output_img = npu.flip_v(output_img)
		
		# Store results
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
