import os
import numpy as np
import cv2
import onnxruntime as ort

# Import the base inference manager
ONNXInferenceManager = mod(f'{op.PyUtils}/onnx_inference_manager').ONNXInferenceManager

# Model Configuration -------------------------------
# Switch between 'general' (256x256) and 'landscape' (256x144) models
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
config = MODEL_CONFIGS['landscape']  # Options: 'general' or 'landscape'


class MediaPipeSelfieSegmentation(ONNXInferenceManager):
	"""MediaPipe Selfie Segmentation inference implementation."""
	
	def get_model_path(self):
		"""Return path to MediaPipe Selfie Segmentation model."""
		model_dir = os.path.join(project.folder, 'data', 'ml', 'models', 'mediapipe-selfie-segmentation')
		model_path = os.path.join(model_dir, config['filename'])
		return model_path
	
	def get_session_options(self):
		"""Return ONNX session options."""
		return ort.SessionOptions()
	
	def on_model_loaded(self, session):
		"""Log model configuration after loading."""
		self.printONNX(f"Config: {config['input_size'][0]}x{config['input_size'][1]}, format: {config['format']}")
	
	def preprocess(self, nA):
		"""Preprocess input for MediaPipe Selfie Segmentation."""
		# Flip and convert color
		# nA = self.npu.flip_v(nA)
		nA = self.npu.rgba_to_rgb(nA)
		nA = self.npu.grayscale_to_rgb(nA)
		
		# MediaPipe models expect BGR format - now doing this in TouchDesigner input
		# nA = cv2.cvtColor(nA, cv2.COLOR_RGB2BGR)
		
		# Resize to model input size (width, height from config)
		input_size = config['input_size']
		if nA.shape[:2] != (input_size[1], input_size[0]):  # shape is (height, width)
			self.printONNX(f"Resizing input from {nA.shape[1]}x{nA.shape[0]} to {input_size[0]}x{input_size[1]}")
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
		
		return input_tensor
	
	def postprocess(self, outputs):
		"""Postprocess MediaPipe segmentation output."""
		# Get the segmentation mask
		mask = outputs[0]
		
		# Remove batch dimension if present
		if mask.shape[0] == 1:
			mask = mask[0]
		
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
		
		return output_img


# Create global instance of the inference manager
inference_manager = MediaPipeSelfieSegmentation()
inference_manager.opPerformance = op('constant_performance')

# TouchDesigner callback wrappers that delegate to the manager
def onSetupParameters(scriptOp):
	return inference_manager.onSetupParameters(scriptOp)


def onPulse(par):
	return inference_manager.onPulse(par)


def onCook(scriptOp):
	return inference_manager.onCook(scriptOp)
