import os

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
KEYPOINT_NAMES = ['eye_r', 'eye_l', 'nose', 'mouth_r', 'mouth_l']
detection_boxes = []  # List of bounding boxes [x1, y1, x2, y2, score]
detection_keypoints = []  # List of keypoint arrays
num_faces_detected = 0  # Number of faces detected in last inference

# Detection config
CONF_THRESHOLD = 0.7  # Confidence threshold for face detections (0.0 - 1.0)

# Tracker config
TRACKER_MAX_AGE = 45     # Max frames to keep a lost track alive
TRACKER_IOU_THRESHOLD = 0.2  # IoU threshold for matching
TRACKER_DIST_THRESHOLD = 5 # Max center-distance (in face-widths) for fallback matching


# ==================== FACE TRACKER ====================

class TrackedFace:
	"""A single tracked face with temporal persistence and head direction estimation."""

	_next_id = 1

	def __init__(self, box, keypoints, score):
		self.track_id = TrackedFace._next_id
		TrackedFace._next_id += 1
		self.box = box            # [x1, y1, x2, y2] normalized 0-1
		self.keypoints = keypoints  # [[x,y], ...] 5 keypoints normalized
		self.score = score
		self.lost_frames = 0
		self.total_frames = 1
		self.velocity = np.zeros(4)  # box delta per frame
		# Head direction (computed from keypoints)
		self.yaw = 0.0    # left/right rotation (-1 to 1)
		self.pitch = 0.0   # up/down rotation (-1 to 1)
		self.roll = 0.0    # tilt rotation (radians)
		self._update_head_direction()

	def update(self, box, keypoints, score):
		old_box = np.array(self.box)
		new_box = np.array(box)
		alpha = 0.3
		self.velocity = alpha * (new_box - old_box) + (1 - alpha) * self.velocity
		self.box = box
		self.keypoints = keypoints
		self.score = score
		self.lost_frames = 0
		self.total_frames += 1
		self._update_head_direction()

	def predict(self):
		"""Mark as lost (no position prediction — face stays at last known position)."""
		self.lost_frames += 1
		self.total_frames += 1
		self.score *= 0.95

	def _update_head_direction(self):
		"""Estimate head yaw, pitch, and roll from facial keypoints.
		
		Keypoints: 0=right_eye, 1=left_eye, 2=nose, 3=right_mouth, 4=left_mouth
		
		Yaw (left/right): Nose position relative to midpoint between eyes.
			If nose is left of center -> looking left (negative).
		Pitch (up/down): Nose vertical position relative to eye-mouth midline.
			If nose is above midline -> looking up (positive in TD coords).
		Roll (tilt): Angle of the line between the two eyes.
		"""
		if not self.keypoints or len(self.keypoints) < 5:
			self.yaw = self.pitch = self.roll = 0.0
			return

		kp = self.keypoints
		eye_r = np.array(kp[0])  # right eye
		eye_l = np.array(kp[1])  # left eye
		nose = np.array(kp[2])   # nose tip
		mouth_r = np.array(kp[3])
		mouth_l = np.array(kp[4])

		# Eye midpoint and mouth midpoint
		eye_mid = (eye_r + eye_l) * 0.5
		mouth_mid = (mouth_r + mouth_l) * 0.5

		# Inter-eye distance (used as normalization scale)
		eye_dist = np.linalg.norm(eye_l - eye_r)
		if eye_dist < 1e-6:
			self.yaw = self.pitch = self.roll = 0.0
			return

		# Yaw: horizontal offset of nose from eye midpoint, normalized by eye distance
		# Positive = looking right (in image space), clamped to [-1, 1]
		self.yaw = np.clip((nose[0] - eye_mid[0]) / eye_dist, -1.0, 1.0)

		# Pitch: vertical offset of nose from midline between eyes and mouth
		# In TD coords (Y up), positive = looking up
		face_mid_y = (eye_mid[1] + mouth_mid[1]) * 0.5
		face_height = abs(eye_mid[1] - mouth_mid[1])
		if face_height > 1e-6:
			self.pitch = np.clip((nose[1] - face_mid_y) / face_height, -1.0, 1.0)
		else:
			self.pitch = 0.0

		# Roll: angle of eye line (radians, positive = tilted clockwise in TD coords)
		dx = eye_l[0] - eye_r[0]
		dy = eye_l[1] - eye_r[1]
		self.roll = float(np.arctan2(dy, dx))


def compute_iou(box_a, box_b):
	"""Compute IoU between two [x1, y1, x2, y2] boxes."""
	x1 = max(box_a[0], box_b[0])
	y1 = max(box_a[1], box_b[1])
	x2 = min(box_a[2], box_b[2])
	y2 = min(box_a[3], box_b[3])
	inter = max(0, x2 - x1) * max(0, y2 - y1)
	area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
	area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
	union = area_a + area_b - inter
	return inter / union if union > 0 else 0.0


class FaceTracker:
	"""Greedy IoU-based face tracker with center-distance fallback for lost tracks."""

	def __init__(self, iou_threshold=TRACKER_IOU_THRESHOLD, max_age=TRACKER_MAX_AGE, dist_threshold=TRACKER_DIST_THRESHOLD):
		self.tracks = []
		self.iou_threshold = iou_threshold
		self.max_age = max_age
		self.dist_threshold = dist_threshold

	def _box_center_and_size(self, box):
		"""Return (cx, cy, w, h) from [x1, y1, x2, y2]."""
		return (
			(box[0] + box[2]) * 0.5,
			(box[1] + box[3]) * 0.5,
			box[2] - box[0],
			box[3] - box[1],
		)

	def update(self, detections):
		"""
		Match new detections to existing tracks via greedy IoU,
		then fall back to center-distance matching for remaining lost tracks.
		detections: list of dicts with 'box', 'keypoints', 'score'
		Returns: list of TrackedFace (active tracks)
		"""
		if not detections:
			for t in self.tracks:
				t.predict()
			self.tracks = [t for t in self.tracks if t.lost_frames <= self.max_age]
			return self.tracks

		unmatched_dets = list(range(len(detections)))
		matched_tracks = set()

		if self.tracks:
			track_boxes = np.array([t.box for t in self.tracks])
			det_boxes = np.array([d['box'] for d in detections])
			# Vectorized IoU matrix
			tb = track_boxes[:, np.newaxis, :]
			db = det_boxes[np.newaxis, :, :]
			inter_x1 = np.maximum(tb[:, :, 0], db[:, :, 0])
			inter_y1 = np.maximum(tb[:, :, 1], db[:, :, 1])
			inter_x2 = np.minimum(tb[:, :, 2], db[:, :, 2])
			inter_y2 = np.minimum(tb[:, :, 3], db[:, :, 3])
			inter = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
			area_t = (track_boxes[:, 2] - track_boxes[:, 0]) * (track_boxes[:, 3] - track_boxes[:, 1])
			area_d = (det_boxes[:, 2] - det_boxes[:, 0]) * (det_boxes[:, 3] - det_boxes[:, 1])
			union = area_t[:, np.newaxis] + area_d[np.newaxis, :] - inter
			iou_matrix = np.where(union > 0, inter / union, 0.0)

			while True:
				if iou_matrix.size == 0:
					break
				i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
				if iou_matrix[i, j] < self.iou_threshold:
					break
				self.tracks[i].update(
					detections[j]['box'],
					detections[j]['keypoints'],
					detections[j]['score']
				)
				matched_tracks.add(i)
				unmatched_dets.remove(j)
				iou_matrix[i, :] = 0
				iou_matrix[:, j] = 0

		# --- Fallback: center-distance matching for unmatched lost tracks ---
		# Only try to match unmatched detections against tracks that are already lost
		if unmatched_dets:
			lost_track_indices = [
				i for i in range(len(self.tracks))
				if i not in matched_tracks and self.tracks[i].lost_frames > 0
			]
			if lost_track_indices:
				for j in list(unmatched_dets):
					det_cx, det_cy, det_w, det_h = self._box_center_and_size(detections[j]['box'])
					det_size = max(det_w, det_h, 1e-6)
					best_dist = float('inf')
					best_i = None
					for i in lost_track_indices:
						t_cx, t_cy, t_w, t_h = self._box_center_and_size(self.tracks[i].box)
						# Normalize distance by average of detection and track size
						avg_size = (det_size + max(t_w, t_h, 1e-6)) * 0.5
						dist = ((det_cx - t_cx)**2 + (det_cy - t_cy)**2)**0.5 / avg_size
						if dist < best_dist:
							best_dist = dist
							best_i = i
					if best_i is not None and best_dist < self.dist_threshold:
						self.tracks[best_i].update(
							detections[j]['box'],
							detections[j]['keypoints'],
							detections[j]['score']
						)
						matched_tracks.add(best_i)
						lost_track_indices.remove(best_i)
						unmatched_dets.remove(j)

		# Age unmatched tracks
		for i, track in enumerate(self.tracks):
			if i not in matched_tracks:
				track.predict()

		# Create new tracks for unmatched detections
		for j in unmatched_dets:
			det = detections[j]
			self.tracks.append(TrackedFace(det['box'], det['keypoints'], det['score']))

		# Prune dead tracks
		self.tracks = [t for t in self.tracks if t.lost_frames <= self.max_age]

		return self.tracks


# Global tracker instance
face_tracker = FaceTracker()
tracked_faces = []  # Current tracked face data for CHOP output

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
	# confidence threshold
	p = page.appendFloat('Confthreshold', label='Confidence Threshold', size=1)
	p[0].default = CONF_THRESHOLD
	p[0].min = 0.0
	p[0].max = 1.0
	p[0].clampMin = True
	p[0].clampMax = True
	scriptOp.par.Confthreshold = CONF_THRESHOLD
	# tracker max age
	p = page.appendInt('Trackermaxage', label='Tracker Max Age (frames)', size=1)
	p[0].default = TRACKER_MAX_AGE
	p[0].min = 1
	p[0].clampMin = True
	scriptOp.par.Trackermaxage = TRACKER_MAX_AGE
	# tracker IoU threshold
	p = page.appendFloat('Trackeriou', label='Tracker IoU Threshold', size=1)
	p[0].default = TRACKER_IOU_THRESHOLD
	p[0].min = 0.0
	p[0].max = 1.0
	p[0].clampMin = True
	p[0].clampMax = True
	scriptOp.par.Trackeriou = TRACKER_IOU_THRESHOLD
	# tracker distance threshold
	p = page.appendFloat('Trackerdist', label='Tracker Distance Threshold', size=1)
	p[0].default = TRACKER_DIST_THRESHOLD
	p[0].min = 0.0
	p[0].clampMin = True
	scriptOp.par.Trackerdist = TRACKER_DIST_THRESHOLD
	return


# called whenever custom pulse parameter is pushed
def onPulse(par):
	if par.name == 'Reloadonnx':
		model = None  # reset the model
	return


# Table DAT output -------------------------------

TABLE_HEADER = [
	'track_id', 'score', 'cx', 'cy', 'w', 'h',
	'x_left', 'x_right', 'y_top', 'y_bottom',
	'yaw', 'pitch', 'roll',
	'eye_r:tx', 'eye_r:ty', 'eye_l:tx', 'eye_l:ty',
	'nose:tx', 'nose:ty', 'mouth_r:tx', 'mouth_r:ty', 'mouth_l:tx', 'mouth_l:ty',
	'vx', 'vy', 'lost_frames', 'total_frames',
]

def write_tracks_to_table():
	"""Write current tracked face data to a Table DAT."""
	tbl = parent().op('table_output')
	if tbl is None:
		return
	tbl.clear()
	tbl.appendRow(TABLE_HEADER)
	for obj in tracked_faces:
		kps = obj.get('keypoints', [[0,0]]*5)
		tbl.appendRow([
			obj['track_id'],
			f"{obj['score']:.3f}",
			f"{obj['cx']:.4f}", f"{obj['cy']:.4f}",
			f"{obj['w']:.4f}", f"{obj['h']:.4f}",
			f"{obj['x_left']:.4f}", f"{obj['x_right']:.4f}",
			f"{obj['y_top']:.4f}", f"{obj['y_bottom']:.4f}",
			f"{obj['yaw']:.4f}", f"{obj['pitch']:.4f}", f"{obj['roll']:.4f}",
			f"{kps[0][0]:.4f}", f"{kps[0][1]:.4f}",
			f"{kps[1][0]:.4f}", f"{kps[1][1]:.4f}",
			f"{kps[2][0]:.4f}", f"{kps[2][1]:.4f}",
			f"{kps[3][0]:.4f}", f"{kps[3][1]:.4f}",
			f"{kps[4][0]:.4f}", f"{kps[4][1]:.4f}",
			f"{obj['vx']:.4f}", f"{obj['vy']:.4f}",
			obj['lost_frames'], obj['total_frames'],
		])


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


def _postprocess_faces(faces, scriptOp=None):
	"""Postprocess raw YuNet detection output into tracked faces with head direction.
	Runs on main thread — no locks needed.
	"""
	global detection_boxes, detection_keypoints, num_faces_detected, last_postprocess_ms, tracked_faces
	
	t0 = time.perf_counter()
	detections = []
	filtered_boxes = []
	filtered_keypoints = []
	
	# Read threshold from custom parameter
	conf_threshold = scriptOp.par.Confthreshold.eval() if scriptOp else CONF_THRESHOLD

	if faces[1] is not None:
		all_faces = faces[1]
		# Vectorized confidence filter
		confidences = all_faces[:, 14]
		valid_mask = confidences >= conf_threshold
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
				box = [
					float(x1_norm[i]),
					float(y2_flip[i]),
					float(x2_norm[i]),
					float(y1_flip[i]),
				]
				score = float(scores[i])
				
				# Vectorized keypoint normalization
				kps_raw = valid_faces[i, 4:14].reshape(5, 2)
				kps_norm = np.empty_like(kps_raw)
				kps_norm[:, 0] = kps_raw[:, 0] * inv_w
				kps_norm[:, 1] = 1.0 - (kps_raw[:, 1] * inv_h)
				kps = kps_norm.tolist()
				
				filtered_boxes.append(box + [score])
				filtered_keypoints.append(kps)
				detections.append({
					'box': box,
					'keypoints': kps,
					'score': score,
				})
	
	detection_boxes = filtered_boxes
	detection_keypoints = filtered_keypoints
	
	# Update tracker with new detections
	active_tracks = face_tracker.update(detections)
	num_faces_detected = len([t for t in active_tracks if t.lost_frames == 0])

	# Build structured data for Table DAT output (like YOLO26)
	tracked_faces = []
	for t in active_tracks:
		cx = (t.box[0] + t.box[2]) * 0.5
		cy = (t.box[1] + t.box[3]) * 0.5
		w = t.box[2] - t.box[0]
		h = t.box[3] - t.box[1]
		tracked_faces.append({
			'track_id': t.track_id,
			'score': t.score,
			'cx': cx, 'cy': cy, 'w': w, 'h': h,
			'x_left': t.box[0],
			'x_right': t.box[2],
			'y_top': t.box[3],
			'y_bottom': t.box[1],
			'yaw': t.yaw, 'pitch': t.pitch, 'roll': t.roll,
			'keypoints': t.keypoints if t.keypoints and len(t.keypoints) >= NUM_KEYPOINTS else [[0,0]]*5,
			'vx': float(t.velocity[0]), 'vy': float(t.velocity[1]),
			'lost_frames': t.lost_frames,
			'total_frames': t.total_frames,
		})

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
			_postprocess_faces(raw_faces, scriptOp)

			# Sync tracker params from custom parameters
			face_tracker.max_age = int(scriptOp.par.Trackermaxage.eval())
			face_tracker.iou_threshold = scriptOp.par.Trackeriou.eval()
			face_tracker.dist_threshold = scriptOp.par.Trackerdist.eval()

			# Write tracking data to Table DAT
			write_tracks_to_table()
			
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
