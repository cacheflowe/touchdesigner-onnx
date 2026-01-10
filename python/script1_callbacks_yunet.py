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

# Face detection results -------------------------------

NUM_KEYPOINTS = 5  # Number of facial keypoints for YuNet (right eye, left eye, nose tip, right mouth corner, left mouth corner)
detection_boxes = []  # List of bounding boxes [x1, y1, x2, y2, score]
detection_keypoints = []  # List of keypoint arrays
num_faces_detected = 0  # Number of faces detected in last inference

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
	# rebuild every frame to handle variable number of faces	
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
	"""Background thread for preprocessing, YuNet inference, and post-processing."""
	global pending_result, is_inferencing, frames_skipped_final, detection_boxes, detection_keypoints, num_faces_detected, cached_input_size
	
	try:
		# Preprocessing - prepare image for face detection
		nA = input_tensor_cache  # This is the raw numpy array from texture
		input_h, input_w = nA.shape[:2]
		
		# Resize for faster inference (coordinates will be scaled back)
		inference_w = int(input_w)
		inference_h = int(input_h)
		
		# Convert from TD format (0-1, RGBA, flipped) to CV format (0-255, BGR)
		# Optimize: combine operations to reduce memory copies
		nA = npu.flip_v(nA)
		nA = (nA[:, :, :3] * 255).astype(np.uint8)  # Convert RGBA to RGB and denormalize in one step
				
		# Convert RGB to BGR for OpenCV (just reverse the color channels)
		img_bgr = nA[:, :, ::-1]  # Faster than cv2.cvtColor
		
		# Only call setInputSize if dimensions changed (expensive operation)
		current_size = (inference_w, inference_h)
		if cached_input_size != current_size:
			model.setInputSize(current_size)
			cached_input_size = current_size
			printONNX(f"Set YuNet input size to: {current_size}")
		
		# Run YuNet detection
		faces = model.detect(img_bgr)
		
		# Parse YuNet output format
		filtered_boxes = []
		filtered_keypoints = []
		
		if faces[1] is not None:
			for face in faces[1]:  # Process all detected faces
				# Extract bbox (x, y, w, h format) and convert to (x1, y1, x2, y2)
				x, y, w, h = face[:4]
				confidence = face[14]
				
				if confidence < 0.3:
					continue  # Skip low-confidence detections
				
				# Convert to normalized coordinates (0-1) in x1,y1,x2,y2 format
				# Note: coordinates are in scaled inference space, normalize directly
				# Flip Y coordinates because we flipped the image vertically for OpenCV
				y1_norm = 1.0 - (y / inference_h)  # Flip Y coordinate
				y2_norm = 1.0 - ((y + h) / inference_h)  # Flip Y coordinate
				
				scaled_box = [
					x / inference_w,
					y2_norm,  # Use flipped Y (smaller value, bottom of bbox)
					(x + w) / inference_w,
					y1_norm,  # Use flipped Y (larger value, top of bbox)
					confidence
				]
				filtered_boxes.append(scaled_box)
				
				# Extract 5 keypoints (right eye, left eye, nose, right mouth, left mouth)
				keypoints_raw = face[4:14].reshape(5, 2)  # Reshape to 5 points x 2 coords
				scaled_kps = []
				for kp in keypoints_raw:
					scaled_kps.append([
						kp[0] / inference_w,
						1.0 - (kp[1] / inference_h)  # Flip Y coordinate
					])
				filtered_keypoints.append(scaled_kps)
		
		# Store results thread-safely
		with inference_lock:
			detection_boxes = filtered_boxes
			detection_keypoints = filtered_keypoints
			num_faces_detected = len(filtered_boxes)
			pending_result = True  # Signal that we have new results
		
	except Exception as e:
		printONNX(f"Inference error: {e}")
		import traceback
		traceback.print_exc()
	finally:
		is_inferencing = False
		frames_skipped_final = frames_skipped


def onCook(scriptOp):
	global session, is_loading, load_error, is_inferencing, pending_result, input_tensor_cache, frames_skipped, frames_skipped_final, num_faces_detected

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

	# Check if we have results from background thread
	with inference_lock:
		if pending_result is not None:
			pending_result = None
			frames_skipped = 0
			# printONNX("Inference thread completed.", frames_skipped_final, "frames were skipped.")
			op('constant_performance').par.const0value = frames_skipped_final
			op('constant_performance').par.const1value = num_faces_detected

			# Update CHOP channels with detection results
			updateChopChannels(scriptOp)

	# If inference is still running, skip this frame
	if is_inferencing:
		frames_skipped += 1
		return
	
	# Capture input on main thread (GPU texture access only)
	try:
		inputTex = op('null_input')
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
