import os
import sys

# import other dependencies now that the path supports it
import time
import threading
import numpy as np
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

# Face detection results -------------------------------

NUM_KEYPOINTS = 5  # Number of facial keypoints for YuNet (right eye, left eye, nose tip, right mouth corner, left mouth corner)
detection_boxes = []  # List of bounding boxes [x1, y1, x2, y2, score]
detection_keypoints = []  # List of keypoint arrays
num_faces_detected = 0  # Number of faces detected in last inference
_prev_num_faces = -1  # Track previous face count to avoid unnecessary CHOP rebuilds

# Timing instrumentation
last_preprocess_ms = 0
last_inference_ms = 0
last_postprocess_ms = 0
_frame_count = 0
_timing_log_interval = 30

# YuNet setup -------------------------------

model = None  # YuNet face detector model
cached_input_size = None  # Cache to avoid unnecessary setInputSize calls
inference_scale = 0.5  # Scale input image (0.5 = half resolution, faster inference)

def printONNX(*args):
	print("[ONNX]", *args)

def loadONNX(scriptOp):
	global model, loading_thread, is_loading

	if is_loading:
		printONNX("Model is already loading...") 
		return

	# Reset model and start loading thread
	model = None
	scriptOp.par.Loadstatus = "loading"
	loading_thread = threading.Thread(target=_load_model_thread)
	loading_thread.daemon = True
	loading_thread.start()


def _load_model_thread():
	global model, is_loading, load_error

	is_loading = True
	load_error = None

	try:
		printONNX('=============================================')
		printONNX("Starting YuNet model loading in background...")

		# Build model path
		model_path = os.path.join(project.folder, 'data', 'ml', 'models', 'yunet', 'face_detection_yunet_2023mar.onnx')
		printONNX("model:", model_path)

		# Create YuNet face detector - this is part of cv2!
		# Parameters: model_path, config_path, input_size, score_threshold, nms_threshold, top_k
		temp_model = cv2.FaceDetectorYN.create(
			model_path,
			"",  # config path (empty for default)
			(256, 256),  # initial input size (will be updated per frame)
			0.5,  # score threshold
			0.3,  # NMS threshold
			5000  # top_k detections
		)
		
		# Only assign to global model when fully loaded
		model = temp_model
		printONNX("YuNet model loaded successfully!")
		printONNX('=============================================')

	except Exception as e:
		load_error = str(e)
		printONNX(f"Error loading YuNet model: {e}")
	finally:
		is_loading = False


def get_loading_status():
	"""Returns status of model loading"""
	if model is not None:
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
		model = None  # reset the model
	return


# CHOP channel management -------------------------------

def buildChopChannels(scriptOp, num_faces):
	"""Build CHOP channels for face detection output dynamically based on number of faces."""
	global _prev_num_faces
	# Skip rebuild if face count hasn't changed
	if num_faces == _prev_num_faces:
		return
	_prev_num_faces = num_faces

	scriptOp.clear()
	
	for i in range(num_faces):
		face_prefix = f"face{i+1}"
		
		# Bounding box channels
		scriptOp.appendChan(f"{face_prefix}/bbox_x1")
		scriptOp.appendChan(f"{face_prefix}/bbox_y1")
		scriptOp.appendChan(f"{face_prefix}/bbox_x2")
		scriptOp.appendChan(f"{face_prefix}/bbox_y2")
		scriptOp.appendChan(f"{face_prefix}/bbox_width")
		scriptOp.appendChan(f"{face_prefix}/bbox_height")
		scriptOp.appendChan(f"{face_prefix}/bbox_center_x")
		scriptOp.appendChan(f"{face_prefix}/bbox_center_y")
		scriptOp.appendChan(f"{face_prefix}/confidence")
		
		# Keypoint channels (5 keypoints with x, y coordinates)
		for kp_idx in range(NUM_KEYPOINTS):
			scriptOp.appendChan(f"{face_prefix}/kp{kp_idx+1}:tx")
			scriptOp.appendChan(f"{face_prefix}/kp{kp_idx+1}:ty")


def updateChopChannels(scriptOp):
	"""Update CHOP channel values with detection results."""
	global detection_boxes, detection_keypoints, num_faces_detected
	
	num_faces = len(detection_boxes)
	num_faces_detected = num_faces
	
	# Rebuild channels if needed
	buildChopChannels(scriptOp, num_faces)
	
	# Update values for all detected faces
	for i in range(num_faces):
		face_prefix = f"face{i+1}"
		
		# We have a detection for this face
		box = detection_boxes[i]
		x1, y1, x2, y2, score = box
		width = x2 - x1
		height = y2 - y1
		center_x = x1 + width * 0.5
		center_y = y1 + height * 0.5
		
		scriptOp[f"{face_prefix}/bbox_x1"][0] = x1
		scriptOp[f"{face_prefix}/bbox_y1"][0] = y1
		scriptOp[f"{face_prefix}/bbox_x2"][0] = x2
		scriptOp[f"{face_prefix}/bbox_y2"][0] = y2
		scriptOp[f"{face_prefix}/bbox_width"][0] = width
		scriptOp[f"{face_prefix}/bbox_height"][0] = height
		scriptOp[f"{face_prefix}/bbox_center_x"][0] = center_x
		scriptOp[f"{face_prefix}/bbox_center_y"][0] = center_y
		scriptOp[f"{face_prefix}/confidence"][0] = score
		
		# Update keypoints if available
		if i < len(detection_keypoints):
			kps = detection_keypoints[i]
			for kp_idx in range(min(NUM_KEYPOINTS, len(kps))):
				if kp_idx < len(kps):
					kp = kps[kp_idx]
					scriptOp[f"{face_prefix}/kp{kp_idx+1}:tx"][0] = kp[0]
					scriptOp[f"{face_prefix}/kp{kp_idx+1}:ty"][0] = kp[1]
				else:
					scriptOp[f"{face_prefix}/kp{kp_idx+1}:tx"][0] = 0.0
					scriptOp[f"{face_prefix}/kp{kp_idx+1}:ty"][0] = 0.0
		else:
			# No keypoints for this face
			for kp_idx in range(NUM_KEYPOINTS):
				scriptOp[f"{face_prefix}/kp{kp_idx+1}:tx"][0] = 0.0
				scriptOp[f"{face_prefix}/kp{kp_idx+1}:ty"][0] = 0.0


def _inference_thread():
	"""Background thread for YuNet inference ONLY.
	Preprocess and postprocess run on the main thread for better performance.
	"""
	global pending_result, is_inferencing, frames_skipped_final
	global last_inference_ms
	
	try:
		# === INFERENCE ONLY ===
		t0 = time.perf_counter()
		faces = model.detect(input_tensor_cache)
		last_inference_ms = (time.perf_counter() - t0) * 1000
		
		# Store raw results thread-safely
		with inference_lock:
			pending_result = faces
	
	except Exception as e:
		printONNX(f"Inference error: {e}")
		import traceback
		traceback.print_exc()
	finally:
		is_inferencing = False
		frames_skipped_final = frames_skipped


def _postprocess_faces(faces):
	"""Postprocess raw YuNet detection output into normalized boxes and keypoints.
	Runs on main thread — no locks needed.
	"""
	global detection_boxes, detection_keypoints, num_faces_detected, last_postprocess_ms
	
	t0 = time.perf_counter()
	filtered_boxes = []
	filtered_keypoints = []
	
	if faces[1] is not None:
		all_faces = faces[1]
		# Vectorized confidence filter
		confidences = all_faces[:, 14]
		valid_mask = confidences >= 0.3
		valid_faces = all_faces[valid_mask]
		
		if len(valid_faces) > 0:
			# Get inference dimensions from cached input
			inference_h, inference_w = input_tensor_cache.shape[:2]
			
			# Vectorized coordinate conversion
			x = valid_faces[:, 0]
			y = valid_faces[:, 1]
			w = valid_faces[:, 2]
			h = valid_faces[:, 3]
			scores = valid_faces[:, 14]
			
			inv_w = 1.0 / inference_w
			inv_h = 1.0 / inference_h
			
			x1_norm = x * inv_w
			y1_flip = 1.0 - (y * inv_h)
			x2_norm = (x + w) * inv_w
			y2_flip = 1.0 - ((y + h) * inv_h)
			
			for i in range(len(valid_faces)):
				filtered_boxes.append([
					float(x1_norm[i]),
					float(y2_flip[i]),
					float(x2_norm[i]),
					float(y1_flip[i]),
					float(scores[i])
				])
				
				# Vectorized keypoint normalization
				kps_raw = valid_faces[i, 4:14].reshape(5, 2)
				kps_norm = np.empty_like(kps_raw)
				kps_norm[:, 0] = kps_raw[:, 0] * inv_w
				kps_norm[:, 1] = 1.0 - (kps_raw[:, 1] * inv_h)
				filtered_keypoints.append(kps_norm.tolist())
	
	detection_boxes = filtered_boxes
	detection_keypoints = filtered_keypoints
	num_faces_detected = len(filtered_boxes)
	last_postprocess_ms = (time.perf_counter() - t0) * 1000


def onCook(scriptOp):
	global session, is_loading, load_error, is_inferencing, pending_result, input_tensor_cache, frames_skipped, frames_skipped_final, num_faces_detected, cached_input_size
	global last_preprocess_ms, _frame_count

	# Update status parameter
	status = get_loading_status()
	scriptOp.par.Loadstatus = status

	# make sure we've loaded the model
	if model is None:
		if not is_loading:
			loadONNX(scriptOp)
		# Return early if model isn't ready yet
		return

	# Check if we have a loading error
	if load_error:
		printONNX(f"Cannot process: {load_error}")
		return

	# Check if we have raw results from background thread
	with inference_lock:
		if pending_result is not None:
			raw_faces = pending_result
			pending_result = None
			frames_skipped = 0
			
			op('constant_performance').par.const0value = frames_skipped_final
			op('constant_performance').par.const1value = num_faces_detected

			# Postprocess on main thread (safe for TD operator access, no locks needed)
			_postprocess_faces(raw_faces)

			# Update CHOP channels with detection results
			updateChopChannels(scriptOp)
			
			# Log timing periodically
			_frame_count += 1
			if _frame_count % _timing_log_interval == 1:
				total = last_preprocess_ms + last_inference_ms + last_postprocess_ms
				fps = 1000.0 / total if total > 0 else 0
				printONNX(
					f"pre={last_preprocess_ms:.1f}ms  "
					f"infer={last_inference_ms:.1f}ms  "
					f"post={last_postprocess_ms:.1f}ms  "
					f"total={total:.1f}ms  "
					f"fps={fps:.1f}  "
					f"faces={num_faces_detected}"
				)

	# If inference is still running, skip this frame
	if is_inferencing:
		frames_skipped += 1
		return
	
	# Preprocess on main thread (GPU texture access is fast here, no copy needed)
	try:
		inputTex = op('null_input')
		nA = inputTex.numpyArray(delayed=True)
		if nA is None:
			return
		
		# Preprocess: fuse flip + RGBA->BGR + denormalize
		# Read directly from TD's staging buffer (cache-warm on main thread)
		t0 = time.perf_counter()
		input_h, input_w = nA.shape[:2]
		img_bgr = (nA[::-1, :, 2::-1] * 255).astype(np.uint8)
		
		# Only call setInputSize if dimensions changed (expensive operation)
		current_size = (int(input_w), int(input_h))
		if cached_input_size != current_size:
			model.setInputSize(current_size)
			cached_input_size = current_size
			printONNX(f"Set YuNet input size to: {current_size}")
		
		input_tensor_cache = img_bgr
		last_preprocess_ms = (time.perf_counter() - t0) * 1000

	except Exception as e:
		printONNX(f"Error capturing input: {e}")
		return
	
	# Start inference in background thread (runs ONLY model.detect)
	is_inferencing = True
	inference_thread = threading.Thread(target=_inference_thread)
	inference_thread.daemon = True
	inference_thread.start()


def onGetCookLevel(scriptOp: scriptCHOP) -> CookLevel:
	"""
	Sets the scriptOp's cook level, the conditions necessary to cause a cook.

	Return one of the following:
		CookLevel.AUTOMATIC - inputs changed and output being used. TD default behavior.
		CookLevel.ON_CHANGE - inputs changed, output used or not.
		CookLevel.WHEN_USED - every frame when output is being used
		CookLevel.ALWAYS - every frame
	"""

	return CookLevel.ALWAYS
