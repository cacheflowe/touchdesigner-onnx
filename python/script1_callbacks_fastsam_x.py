import os
import numpy as np
import cv2
import onnxruntime as ort

# Import the base inference manager
ONNXInferenceManager = mod(
	f'{op.PyUtils}/onnx_inference_manager').ONNXInferenceManager

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
# Only detect people; add more like: ['person', 'car', 'dog']
CLASSES_TO_DETECT = ['person']


class FastSAMInference(ONNXInferenceManager):
	"""FastSAM (Fast Segment Anything Model) inference implementation."""

	def get_model_path(self):
		"""Return path to FastSAM model."""
		model_dir = os.path.join(project.folder, 'data', 'ml', 'models', 'fastsam-x')
		return os.path.join(model_dir, 'model.onnx')

	def get_session_options(self):
		"""Return ONNX session options."""
		return ort.SessionOptions()

	def preprocess(self, nA):
		"""Preprocess input for FastSAM (YOLO-based model)."""
		nA = self.npu.flip_v(nA)
		nA = self.npu.rgba_to_rgb(nA)
		nA = self.npu.grayscale_to_rgb(nA)

		# Store original size for later
		self.original_h, self.original_w = nA.shape[:2]

		# Resize to model input size (640x640 for FastSAM - required by model)
		input_size = 640
		if nA.shape[:2] != (input_size, input_size):
			nA = cv2.resize(nA, (input_size, input_size),
			                interpolation=cv2.INTER_NEAREST)  # NEAREST is faster

		# YOLO expects [batch, channels, height, width] format
		input_tensor = nA.transpose(2, 0, 1)  # HWC to CHW
		input_tensor = np.expand_dims(
			input_tensor, axis=0).astype('float32')  # Add batch

		return input_tensor

	def postprocess(self, outputs):
		"""Postprocess FastSAM output."""
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
			self.printONNX("Unexpected number of outputs:", len(outputs))
			output_img = np.zeros(
				(self.original_h, self.original_w, 3), dtype=np.float32)
			return self.npu.flip_v(output_img)

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
			masks = np.matmul(valid_coeffs, mask_protos.reshape(
				32, -1)).reshape(-1, 160, 160)

			# Apply sigmoid to get probabilities
			masks = 1 / (1 + np.exp(-masks))

			# Threshold masks to get binary masks
			mask_threshold = 0.25
			binary_masks = (masks > mask_threshold).astype(np.float32)

			# Filter by mask area - only keep larger segments
			min_area_ratio = 0.03  # Minimum 3% of image area - higher = faster
			max_masks = 5  # Limit to top 5 largest masks for speed

			# Vectorized area calculation
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

			self.printONNX(f"Found {len(filtered_masks)} objects after filtering")

			# Create a color palette for different segments
			SEGMENT_COLORS = [
				(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0, 0.0),
				(1.0, 0.0, 1.0), (0.0, 1.0, 1.0), (1.0, 0.5, 0.0), (0.5, 0.0, 1.0),
				(0.0, 1.0, 0.5), (1.0, 0.0, 0.5), (0.5, 1.0, 0.0), (0.0, 0.5, 1.0),
			]

			# Create colored output - start with black background
			output_img = np.zeros((160, 160, 3), dtype=np.float32)

			# Vectorized color application
			for i, mask in enumerate(filtered_masks):
				# Cycle through color palette
				color = SEGMENT_COLORS[i % len(SEGMENT_COLORS)]
				# Apply color directly using broadcasting
				for c in range(3):
					output_img[:, :, c] = np.maximum(output_img[:, :, c], mask * color[c])

			# Resize to original dimensions using faster interpolation
			output_img = cv2.resize(
				output_img, (self.original_w, self.original_h), interpolation=cv2.INTER_NEAREST)
		else:
			# No detections above threshold
			output_img = np.zeros(
				(self.original_h, self.original_w, 3), dtype=np.float32)

		output_img = self.npu.flip_v(output_img)
		return output_img


# Create global instance of the inference manager
inference_manager = FastSAMInference()
inference_manager.opPerformance = op('constant_performance')

# TouchDesigner callback wrappers that delegate to the manager


def onSetupParameters(scriptOp):
	return inference_manager.onSetupParameters(scriptOp)


def onPulse(par):
	return inference_manager.onPulse(par)


def onCook(scriptOp):
	return inference_manager.onCook(scriptOp)
