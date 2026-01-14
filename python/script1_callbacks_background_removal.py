import os
import numpy as np
import cv2

# Import the base inference manager
ONNXInferenceManager = mod(f'{op.PyUtils}/onnx_inference_manager').ONNXInferenceManager


class BackgroundRemovalInference(ONNXInferenceManager):
	"""Person U2Net-P background removal inference implementation."""
	
	def get_model_path(self):
		"""Return path to Person U2Net-P model."""
		return os.path.join(project.folder, 'data', 'ml', 'models', 'person-u2netp', 'person-u2netp.onnx')
	
	def preprocess(self, nA):
		"""Preprocess input for Person U2Net-P model."""
		nA = self.npu.flip_v(nA)
		nA = self.npu.rgba_to_rgb(nA)
		nA = self.npu.grayscale_to_rgb(nA)
		
		# TouchDesigner textures are already 0-1 float, so denormalize to 0-255
		nA = self.npu.denormalize_td_image(nA)

		# Resize to model input size
		if nA.shape[:2] != (384, 384):
			nA = cv2.resize(nA, (384, 384), interpolation=cv2.INTER_CUBIC)
		
		# PyTorch standardization: (x / 255 - mean) / std
		nA = self.npu.imagenet_normalize(nA)
		
		# Transpose to CHW format and add batch dimension
		input_tensor = nA.transpose(2, 0, 1)
		input_tensor = input_tensor.reshape(-1, 3, 384, 384).astype('float32')
		
		return input_tensor
	
	def postprocess(self, outputs):
		"""Postprocess Person U2Net-P output."""
		# Get depth map output
		depth_map = outputs[0]
		
		# Post-processing (squeeze and normalize)
		output_img = np.squeeze(depth_map)
		
		# Min-max normalization to 0-255 range
		min_value = np.min(output_img)
		max_value = np.max(output_img)
		output_img = (output_img - min_value) / (max_value - min_value)
		output_img *= 255.0
		output_img = output_img.astype(np.uint8)
		
		# Convert grayscale to RGB (stack 3 channels)
		output_img = np.stack([output_img, output_img, output_img], axis=-1)
		
		# Convert back to 0-1 float for TouchDesigner
		output_img = output_img.astype(np.float32) / 255.0
		output_img = self.npu.flip_v(output_img)
		
		return output_img


# Create global instance of the inference manager
inference_manager = BackgroundRemovalInference()
inference_manager.opPerformance = op('constant_performance')


# TouchDesigner callback wrappers that delegate to the manager
def onSetupParameters(scriptOp):
	return inference_manager.onSetupParameters(scriptOp)


def onPulse(par):
	return inference_manager.onPulse(par)


def onCook(scriptOp):
	return inference_manager.onCook(scriptOp)
