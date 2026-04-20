import os
import numpy as np
import cv2

# custom util imports
onnx_util = mod(f'{op.PyUtils}/onnx_util')
npu = mod(f'{op.PyUtils}/numpy_util')

# Import the base inference manager
ONNXInferenceManager = mod(f'{op.PyUtils}/onnx_inference_manager').ONNXInferenceManager

# COCO class names (80 classes, used by YOLO26)
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

# Class-specific colors (BGR 0-255 for cv2 drawing)
CLASS_COLORS_BGR = {
	'person': (0, 255, 0),        # Green
	'car': (0, 0, 255),           # Red
	'bicycle': (255, 0, 0),       # Blue
	'dog': (0, 255, 255),         # Yellow
	'cat': (0, 128, 255),         # Orange
	'chair': (255, 0, 128),       # Purple
	'bottle': (255, 255, 0),      # Cyan
}
DEFAULT_COLOR_BGR = (255, 255, 255)  # White fallback

# ==================== CONFIGURATION ====================
# Classes to detect (empty list = detect ALL 80 COCO classes)
CLASSES_TO_DETECT = ['person']  # e.g. ['person', 'car', 'dog']

# Which model variant to use: 'yolo26n' (faster) or 'yolo26s' (more accurate)
# NOTE: For best performance, export directly from Ultralytics rather than using
# onnx-community HuggingFace models. The HF models use a DETR transformer decoder
# which is much slower. To export the fast anchor-based model:
#
#   pip install ultralytics
#   python -c "from ultralytics import YOLO; YOLO('yolo26n.pt').export(format='onnx', imgsz=640, simplify=True)"
#
# Then place yolo26n.onnx in data/ml/models/yolo26/
# The script auto-detects DETR vs YOLO output format.
MODEL_VARIANT = 'yolo26n'

# Confidence threshold for detections (0.0 - 1.0)
CONF_THRESHOLD = 0.5

# Max detections to keep after NMS
MAX_DETECTIONS = 50

# IoU threshold for NMS (lower = more aggressive suppression)
NMS_IOU_THRESHOLD = 0.45

# Tracker: max frames to keep a lost track alive
TRACKER_MAX_AGE = 30

# Tracker: IoU threshold for matching detections to existing tracks
TRACKER_IOU_THRESHOLD = 0.3

# Draw bounding boxes on the output image?
DRAW_BOXES = False


# ==================== SIMPLE IoU TRACKER ====================

class TrackedObject:
	"""A single tracked object with temporal persistence."""

	_next_id = 1

	def __init__(self, box, class_id, score):
		self.track_id = TrackedObject._next_id
		TrackedObject._next_id += 1
		self.box = box          # [x1, y1, x2, y2] normalized 0-1
		self.class_id = class_id
		self.score = score
		self.lost_frames = 0    # frames since last matched
		self.total_frames = 1   # total frames this track has existed
		self.velocity = np.array([0.0, 0.0, 0.0, 0.0])  # box delta per frame

	def update(self, box, class_id, score):
		old_box = np.array(self.box)
		new_box = np.array(box)
		# Smooth velocity with exponential moving average
		alpha = 0.3
		self.velocity = alpha * (new_box - old_box) + (1 - alpha) * self.velocity
		self.box = box
		self.class_id = class_id
		self.score = score
		self.lost_frames = 0
		self.total_frames += 1

	def predict(self):
		"""Predict next position using velocity (for unmatched frames)."""
		self.lost_frames += 1
		self.total_frames += 1
		self.box = (np.array(self.box) + self.velocity).tolist()
		self.score *= 0.95  # Decay confidence when unmatched


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


def compute_iou_vectorized(box, boxes):
	"""Compute IoU between one box and an array of boxes. All [x1,y1,x2,y2]."""
	x1 = np.maximum(box[0], boxes[:, 0])
	y1 = np.maximum(box[1], boxes[:, 1])
	x2 = np.minimum(box[2], boxes[:, 2])
	y2 = np.minimum(box[3], boxes[:, 3])
	inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
	area_a = (box[2] - box[0]) * (box[3] - box[1])
	areas_b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
	union = area_a + areas_b - inter
	return np.where(union > 0, inter / union, 0.0)


class SimpleTracker:
	"""Greedy IoU-based multi-object tracker with temporal persistence."""

	def __init__(self, iou_threshold=TRACKER_IOU_THRESHOLD, max_age=TRACKER_MAX_AGE):
		self.tracks = []
		self.iou_threshold = iou_threshold
		self.max_age = max_age

	def update(self, detections):
		"""
		Match new detections to existing tracks via greedy IoU.
		detections: list of dicts with 'box' [x1,y1,x2,y2], 'class_id', 'score'
		Returns: list of TrackedObject (active tracks)
		"""
		if not detections:
			# No detections: age all tracks
			for t in self.tracks:
				t.predict()
			self.tracks = [t for t in self.tracks if t.lost_frames <= self.max_age]
			return self.tracks

		# Build IoU matrix between existing tracks and new detections
		unmatched_dets = list(range(len(detections)))
		matched_tracks = set()

		if self.tracks:
			# Vectorized IoU matrix computation
			track_boxes = np.array([t.box for t in self.tracks])  # (T, 4)
			det_boxes = np.array([d['box'] for d in detections])  # (D, 4)
			# Broadcast IoU: (T, 1, 4) vs (1, D, 4)
			tb = track_boxes[:, np.newaxis, :]  # (T, 1, 4)
			db = det_boxes[np.newaxis, :, :]    # (1, D, 4)
			inter_x1 = np.maximum(tb[:, :, 0], db[:, :, 0])
			inter_y1 = np.maximum(tb[:, :, 1], db[:, :, 1])
			inter_x2 = np.minimum(tb[:, :, 2], db[:, :, 2])
			inter_y2 = np.minimum(tb[:, :, 3], db[:, :, 3])
			inter = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
			area_t = (track_boxes[:, 2] - track_boxes[:, 0]) * (track_boxes[:, 3] - track_boxes[:, 1])
			area_d = (det_boxes[:, 2] - det_boxes[:, 0]) * (det_boxes[:, 3] - det_boxes[:, 1])
			union = area_t[:, np.newaxis] + area_d[np.newaxis, :] - inter
			iou_matrix = np.where(union > 0, inter / union, 0.0)  # (T, D)

			# Greedy matching: pick highest IoU pairs
			while True:
				if iou_matrix.size == 0:
					break
				i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
				if iou_matrix[i, j] < self.iou_threshold:
					break
				# Match track i to detection j
				self.tracks[i].update(
					detections[j]['box'],
					detections[j]['class_id'],
					detections[j]['score']
				)
				matched_tracks.add(i)
				unmatched_dets.remove(j)
				# Zero out this row and column
				iou_matrix[i, :] = 0
				iou_matrix[:, j] = 0

		# Age unmatched tracks
		for i, track in enumerate(self.tracks):
			if i not in matched_tracks:
				track.predict()

		# Create new tracks for unmatched detections
		for j in unmatched_dets:
			det = detections[j]
			self.tracks.append(TrackedObject(det['box'], det['class_id'], det['score']))

		# Prune dead tracks
		self.tracks = [t for t in self.tracks if t.lost_frames <= self.max_age]

		return self.tracks


# ==================== YOLO26 OBJECT DETECTION ====================

class YOLO26ObjectDetectionInference(ONNXInferenceManager):
	"""YOLO26 Object Detection inference with temporal tracking.

	Supports onnx-community HuggingFace models (DETR-style: 300 queries)
	and standard Ultralytics ONNX exports (anchor-based: 8400 candidates).
	The output format is auto-detected at model load time.
	"""

	def __init__(self):
		super().__init__()
		self.opOutputTableDAT = parent().op('table_output')  # Optional Table DAT for structured output
		self.output_format = None  # 'detr' or 'yolo' - detected at load
		self.num_classes = 80
		self.conf_threshold = CONF_THRESHOLD  # Will be overridden by custom par
		self.tracker = SimpleTracker()
		# Structured tracking data exposed for CHOP consumption
		# Each entry: {track_id, class_id, class_name, score, cx, cy, w, h, x_left, x_right, y_top, y_bottom, x1, y1, x2, y2, vx, vy, lost_frames, total_frames}
		self.tracked_objects = []
		self.pending_table_update = False  # Flag for main-thread table flush
		# Pre-allocated buffers (lazily sized)
		self._output_buf = None
		self._output_buf_shape = None
		self._input_tensor_buf = None   # pre-allocated NCHW input buffer
		self._input_buf_shape = None
		# Cache target class IDs
		self._target_ids_array = np.array([idx for idx, name in COCO_CLASSES.items() if name in CLASSES_TO_DETECT], dtype=np.intp) if CLASSES_TO_DETECT else None

	def onSetupParameters(self, scriptOp):
		"""Add YOLO26-specific parameters alongside base class params."""
		super().onSetupParameters(scriptOp)
		page = scriptOp.appendCustomPage('YOLO26')
		p = page.appendFloat('Confthreshold', label='Confidence Threshold', size=1)
		p[0].default = CONF_THRESHOLD
		p[0].min = 0.0
		p[0].max = 1.0
		p[0].clampMin = True
		p[0].clampMax = True
		scriptOp.par.Confthreshold = CONF_THRESHOLD

	def get_model_path(self):
		"""Return path to YOLO26 detection model."""
		# Models from:
		# - https://huggingface.co/onnx-community/yolo26n-ONNX/tree/main/onnx
		# - https://huggingface.co/onnx-community/yolo26s-ONNX/tree/main/onnx
		model_dir = os.path.join(project.folder, 'data', 'ml', 'models', 'yolo26')
		return os.path.join(model_dir, f'{MODEL_VARIANT}.onnx')

	def on_model_loaded(self, session):
		"""Inspect model outputs to determine format (DETR vs traditional YOLO)."""
		outputs = session.get_outputs()
		self.printONNX(f"YOLO26 model outputs ({len(outputs)}):")
		for i, o in enumerate(outputs):
			self.printONNX(f"  [{i}] name='{o.name}' shape={o.shape} type={o.type}")

		inputs = session.get_inputs()
		for i, inp in enumerate(inputs):
			self.printONNX(f"  input[{i}] name='{inp.name}' shape={inp.shape} type={inp.type}")

		# Log active execution providers (critical for performance diagnosis)
		active = session.get_providers()
		self.printONNX(f"Active providers: {active}")
		if 'CUDAExecutionProvider' not in active:
			self.printONNX("WARNING: Running on CPU only! CUDA provider not available.")
			self.printONNX("  Install onnxruntime-gpu or check CUDA/cuDNN compatibility.")

		# Auto-detect output format
		if len(outputs) == 2:
			# DETR-style: logits [1, 300, 80] + pred_boxes [1, 300, 4]
			self.output_format = 'detr'
			self.printONNX("Detected DETR-style output format (300 queries)")
		elif len(outputs) == 1:
			shape = outputs[0].shape
			# Traditional YOLO: [1, 84, 8400] or [1, 8400, 84]
			if shape and len(shape) == 3:
				if shape[1] == 84 or shape[2] == 84:
					self.output_format = 'yolo'
					self.printONNX(f"Detected traditional YOLO output format {shape}")
				else:
					self.output_format = 'yolo'
					self.printONNX(f"Assuming YOLO format for shape {shape}")
			else:
				self.output_format = 'yolo'
				self.printONNX(f"Unknown shape {shape}, defaulting to YOLO format")
		else:
			self.output_format = 'detr'
			self.printONNX(f"Unknown output count {len(outputs)}, trying DETR format")

	def preprocess(self, nA):
		"""Preprocess input for YOLO26 detection model.
		Assumes TD has already resized input to the model's expected dimensions (e.g. 640x640).
		"""
		self.original_h, self.original_w = nA.shape[:2]
		num_channels = nA.shape[2] if len(nA.shape) == 3 else 1

		if num_channels >= 3:
			h, w = self.original_h, self.original_w
			needed = (1, 3, h, w)
			# Allocate buffer only when dimensions change
			if self._input_buf_shape != needed:
				self._input_tensor_buf = np.empty(needed, dtype=np.float32)
				self._input_buf_shape = needed
			# Flip vertically + RGB + CHW copy into pre-allocated buffer
			flipped = nA[::-1, :, :3]  # view, no alloc
			self._input_tensor_buf[0, 0] = flipped[:, :, 0]
			self._input_tensor_buf[0, 1] = flipped[:, :, 1]
			self._input_tensor_buf[0, 2] = flipped[:, :, 2]
		else:
			nA = self.npu.flip_v(nA)
			nA = self.npu.grayscale_to_rgb(nA)
			self._input_tensor_buf = np.ascontiguousarray(nA.transpose(2, 0, 1)[np.newaxis], dtype=np.float32)
			self._input_buf_shape = self._input_tensor_buf.shape

		return self._input_tensor_buf

	def _parse_detr_outputs(self, outputs):
		"""Parse DETR-style outputs: logits + pred_boxes (onnx-community format)."""
		logits = outputs[0][0]     # (300, 80) raw logits
		pred_boxes = outputs[1][0] # (300, 4)  normalized [cx, cy, w, h]

		# Sigmoid to get class probabilities (in-place to avoid allocation)
		np.negative(logits, out=logits)
		np.exp(logits, out=logits)
		np.add(1.0, logits, out=logits)
		np.reciprocal(logits, out=logits)  # logits now contains sigmoid scores

		class_ids = np.argmax(logits, axis=1)    # (300,)
		confidences = np.max(logits, axis=1)     # (300,)

		# Convert [cx, cy, w, h] normalized -> [x1, y1, x2, y2] normalized (in-place)
		half_w = pred_boxes[:, 2] * 0.5
		half_h = pred_boxes[:, 3] * 0.5
		cx = pred_boxes[:, 0]
		cy = pred_boxes[:, 1]
		boxes_xyxy = np.column_stack([cx - half_w, cy - half_h, cx + half_w, cy + half_h])

		return boxes_xyxy, class_ids, confidences

	def _parse_yolo_outputs(self, outputs):
		"""Parse traditional YOLO output: [1, 84, N] or [1, N, 84]."""
		pred = outputs[0][0]  # (84, N) or (N, 84)

		# Determine orientation: 84 classes+boxes dimension
		if pred.shape[0] == 4 + self.num_classes:
			# (84, N) -> transpose to (N, 84)
			pred = pred.T
		# Now pred is (N, 84): first 4 = xywh, rest = class scores

		boxes_xywh = pred[:, :4]  # (N, 4) center_x, center_y, w, h in pixels (input space)
		class_scores = pred[:, 4:]  # (N, 80)

		class_ids = np.argmax(class_scores, axis=1)
		confidences = np.max(class_scores, axis=1)

		# Convert xywh (pixel space) -> xyxy normalized 0-1
		input_h, input_w = self.original_h, self.original_w
		cx, cy, w, h = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
		x1 = (cx - w / 2) / input_w
		y1 = (cy - h / 2) / input_h
		x2 = (cx + w / 2) / input_w
		y2 = (cy + h / 2) / input_h
		boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

		return boxes_xyxy, class_ids, confidences

	def _nms(self, boxes, scores, iou_threshold=NMS_IOU_THRESHOLD):
		"""Simple NMS on [x1,y1,x2,y2] boxes. Returns indices to keep."""
		if len(boxes) == 0:
			return []
		order = scores.argsort()[::-1]
		keep = []
		while len(order) > 0:
			i = order[0]
			keep.append(i)
			if len(order) == 1:
				break
			ious = compute_iou_vectorized(boxes[i], boxes[order[1:]])
			remaining = np.where(ious < iou_threshold)[0]
			order = order[remaining + 1]
		return keep

	def postprocess(self, outputs):
		"""Postprocess YOLO26 detection outputs.

		Parses model output (auto-detected format), applies class filtering,
		confidence thresholding, NMS, tracking, and draws bounding boxes.
		"""
		# Parse outputs based on detected format
		if self.output_format == 'detr':
			boxes_xyxy, class_ids, confidences = self._parse_detr_outputs(outputs)
		else:
			boxes_xyxy, class_ids, confidences = self._parse_yolo_outputs(outputs)

		# Read threshold from custom parameter (updated each frame)
		self.conf_threshold = self.scriptOp.par.Confthreshold.eval() if self.scriptOp else CONF_THRESHOLD

		# Confidence filter
		valid = confidences > self.conf_threshold

		# Class filter (vectorized)
		if self._target_ids_array is not None and len(self._target_ids_array) > 0:
			valid &= np.isin(class_ids, self._target_ids_array)

		boxes_xyxy = boxes_xyxy[valid]
		class_ids = class_ids[valid]
		confidences = confidences[valid]

		# Clip boxes to [0, 1]
		boxes_xyxy = np.clip(boxes_xyxy, 0.0, 1.0)

		# Flip Y-axis for TouchDesigner (model uses top-down, TD uses bottom-up)
		boxes_xyxy[:, 1], boxes_xyxy[:, 3] = 1.0 - boxes_xyxy[:, 3], 1.0 - boxes_xyxy[:, 1]

		# NMS
		if len(boxes_xyxy) > 0:
			keep = self._nms(boxes_xyxy, confidences)
			boxes_xyxy = boxes_xyxy[keep][:MAX_DETECTIONS]
			class_ids = class_ids[keep][:MAX_DETECTIONS]
			confidences = confidences[keep][:MAX_DETECTIONS]

		# Build detection list for tracker
		detections = []
		for i in range(len(boxes_xyxy)):
			detections.append({
				'box': boxes_xyxy[i].tolist(),
				'class_id': int(class_ids[i]),
				'score': float(confidences[i]),
			})

		# Update tracker (runs on main thread, no lock needed)
		active_tracks = self.tracker.update(detections)

		# Build structured data for CHOP output (filter out decayed tracks)
		self.tracked_objects = []
		for t in active_tracks:
			if t.score < self.conf_threshold:
				continue
			cx = (t.box[0] + t.box[2]) / 2
			cy = (t.box[1] + t.box[3]) / 2
			w = t.box[2] - t.box[0]
			h = t.box[3] - t.box[1]
			self.tracked_objects.append({
				'track_id': t.track_id,
				'class_id': t.class_id,
				'class_name': COCO_CLASSES.get(t.class_id, 'unknown'),
				'score': t.score,
				'cx': cx, 'cy': cy, 'w': w, 'h': h,
				'x_left': t.box[0],
				'x_right': t.box[2],
				'y_top': t.box[3],     # top edge of bbox (TD coords)
				'y_bottom': t.box[1],  # bottom edge of bbox (TD coords)
				'vx': float(t.velocity[0]), 'vy': float(t.velocity[1]),
				'lost_frames': t.lost_frames,
				'total_frames': t.total_frames,
			})

		# Draw output image
		if DRAW_BOXES:
			output_img = self.npu.flip_v(self.draw_tracked_boxes())
		else:
			# Black frame — no need to zero or flip each frame, just reuse static buffer
			needed_shape = (self.original_h, self.original_w, 3)
			if self._output_buf is None or self._output_buf_shape != needed_shape:
				self._output_buf = np.zeros(needed_shape, dtype=np.float32)
				self._output_buf_shape = needed_shape
			output_img = self._output_buf

		# Flag that we have new tracking data to flush on main thread
		self.pending_table_update = True

		return output_img

	def draw_tracked_boxes(self):
		"""Render bounding boxes for tracked objects onto a blank image.
		Returns an RGB float32 (0-1) image at original resolution."""
		output_img = np.zeros((self.original_h, self.original_w, 3), dtype=np.float32)

		if not self.tracked_objects:
			return output_img

		# Work in uint8 for cv2 drawing, then convert back
		draw_img = np.zeros((self.original_h, self.original_w, 3), dtype=np.uint8)

		for obj in self.tracked_objects:
			if obj['lost_frames'] > 0 and obj['score'] < self.conf_threshold * 0.5:
				continue  # Skip faded-out unmatched tracks

			px1 = int(obj['x_left'] * self.original_w)
			py1 = int(obj['y_bottom'] * self.original_h)
			px2 = int(obj['x_right'] * self.original_w)
			py2 = int(obj['y_top'] * self.original_h)

			class_name = obj['class_name']
			color = CLASS_COLORS_BGR.get(class_name, DEFAULT_COLOR_BGR)

			# Dim color if track is unmatched (age > 0)
			if obj['lost_frames'] > 0:
				fade = max(0.3, 1.0 - obj['lost_frames'] / TRACKER_MAX_AGE)
				color = tuple(int(c * fade) for c in color)

			thickness = 2
			cv2.rectangle(draw_img, (px1, py1), (px2, py2), color, thickness)

			label = f"#{obj['track_id']} {class_name} {obj['score']:.0%}"
			font_scale = 0.5
			(tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
			cv2.rectangle(draw_img, (px1, py1 - th - 6), (px1 + tw + 4, py1), color, -1)
			cv2.putText(draw_img, label, (px1 + 2, py1 - 4),
				cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)

		# Convert BGR uint8 -> RGB float32 (0-1)
		return cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

	# ==================== CHOP TRACKING OUTPUT ====================
	# To output tracking data to a CHOP, use a Script CHOP DAT with:
	#
	#   mgr = op('script1').module.inference_manager
	#   tracks = mgr.tracked_objects  # list of dicts
	#
	# Each dict contains: track_id, class_id, class_name, score,
	#   cx, cy, w, h, x_left, x_right, y_top, y_bottom, vx, vy, lost_frames, total_frames
	#
	# For a Table DAT approach, call write_tracks_to_table() from a
	# Script DAT or Execute DAT each frame.

	def write_tracks_to_table(self):
		"""Helper to write current tracking data to a Table DAT.
		Call from an Execute DAT's onFrameStart or a Timer callback."""
		tbl = self.opOutputTableDAT
		if tbl is None:
			return
		tbl.clear()
		tbl.appendRow(['track_id', 'class_id', 'class_name', 'score',
					'cx', 'cy', 'w', 'h',
					'x_left', 'x_right', 'y_top', 'y_bottom',
					'vx', 'vy', 'lost_frames', 'total_frames'])
		for obj in self.tracked_objects:
			tbl.appendRow([
				obj['track_id'], obj['class_id'], obj['class_name'],
				f"{obj['score']:.3f}",
				f"{obj['cx']:.4f}", f"{obj['cy']:.4f}",
				f"{obj['w']:.4f}", f"{obj['h']:.4f}",
				f"{obj['x_left']:.4f}", f"{obj['x_right']:.4f}",
				f"{obj['y_top']:.4f}", f"{obj['y_bottom']:.4f}",
				f"{obj['vx']:.4f}", f"{obj['vy']:.4f}",
				obj['lost_frames'], obj['total_frames'],
			])


# Create global instance
inference_manager = YOLO26ObjectDetectionInference()
inference_manager.opPerformance = op('constant_performance')

# TouchDesigner callback wrappers that delegate to the manager
def onSetupParameters(scriptOp):
	return inference_manager.onSetupParameters(scriptOp)


def onPulse(par):
	return inference_manager.onPulse(par)


def onCook(scriptOp):
	# Run base manager cook (handles model loading, inference dispatch, copyNumpyArray)
	inference_manager.onCook(scriptOp)

	# Optionally draw boxes on main thread to avoid threading issues with OpenCV (if enabled)
	global DRAW_BOXES
	DRAW_BOXES = parent().par.Drawdebug.eval() == 1

	# Flush tracking data to Table DAT on main thread (safe for TD operator access)
	if inference_manager.pending_table_update:
		inference_manager.pending_table_update = False
		inference_manager.write_tracks_to_table()


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
