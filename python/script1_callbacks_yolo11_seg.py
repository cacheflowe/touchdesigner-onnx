import os
import numpy as np
import cv2
import onnxruntime as ort

# custom util imports
onnx_util = mod(f'{op.PyUtils}/onnx_util')
npu = mod(f'{op.PyUtils}/numpy_util')

# Import the base inference manager
ONNXInferenceManager = mod(f'{op.PyUtils}/onnx_inference_manager').ONNXInferenceManager

# COCO class names (YOLO11 uses COCO dataset classes)
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


class YOLO11SegmentationInference(ONNXInferenceManager):
	"""YOLO11 Instance Segmentation inference implementation."""
	
	def get_model_path(self):
		"""Return path to YOLO11 segmentation model."""
		model_dir = os.path.join(project.folder, 'data', 'ml', 'models', 'yolo11-seg')
		return os.path.join(model_dir, 'yolo11s-seg.onnx')
	
	def get_session_options(self):
		"""Return ONNX session options."""
		return ort.SessionOptions()
	
	def preprocess(self, nA):
		"""Preprocess input for YOLO11-seg model."""
		nA = self.npu.flip_v(nA)
		nA = self.npu.rgba_to_rgb(nA)
		nA = self.npu.grayscale_to_rgb(nA)
		
		# Store original size for later
		self.original_h, self.original_w = nA.shape[:2]

		# Resize to model input size (640x640 for YOLO11)
		input_size = 640
		if nA.shape[:2] != (input_size, input_size):
			nA = cv2.resize(nA, (input_size, input_size), interpolation=cv2.INTER_NEAREST)
			
		# YOLO expects [batch, channels, height, width] format
		input_tensor = nA.transpose(2, 0, 1)  # HWC to CHW
		input_tensor = np.expand_dims(input_tensor, axis=0).astype('float32')  # Add batch
		
		return input_tensor
	
	def postprocess(self, outputs):
		"""Postprocess YOLO11-seg output."""
		# YOLO11-seg outputs: 
		# output0: [1, 116, 8400] - combined detections (4 box coords + 80 classes + 32 mask coeffs)
		# output1: [1, 32, 160, 160] - mask prototypes
		
		if len(outputs) != 2:
			self.printONNX(f"Unexpected number of outputs: {len(outputs)}")
			output_img = np.zeros((self.original_h, self.original_w, 3), dtype=np.float32)
			return self.npu.flip_v(output_img)
		
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
			valid_coeffs = mask_coeffs[valid_indices]
			valid_classes = class_ids[valid_indices]
			valid_scores = confidences[valid_indices]
			
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
			
			# Sort by score (highest first) and take top N
			if len(filtered_masks) > max_masks:
				top_indices = np.argsort(filtered_scores)[-max_masks:]
				filtered_masks = filtered_masks[top_indices]
				filtered_classes = filtered_classes[top_indices]
			
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
			
			# Resize to original dimensions using smoother interpolation
			output_img = cv2.resize(output_img, (self.original_w, self.original_h), interpolation=cv2.INTER_CUBIC)
		else:
			# No detections
			output_img = np.zeros((self.original_h, self.original_w, 3), dtype=np.float32)
		
		output_img = self.npu.flip_v(output_img)
		return output_img


# Create global instance of the inference manager
inference_manager = YOLO11SegmentationInference()
inference_manager.opPerformance = op('constant_performance')

# TouchDesigner callback wrappers that delegate to the manager
def onSetupParameters(scriptOp):
	return inference_manager.onSetupParameters(scriptOp)


def onPulse(par):
	return inference_manager.onPulse(par)


def onCook(scriptOp):
	return inference_manager.onCook(scriptOp)
