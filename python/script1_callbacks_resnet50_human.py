import os
import numpy as np

# Import the base inference manager
ONNXInferenceManager = mod(f'{op.PyUtils}/onnx_inference_manager').ONNXInferenceManager


class ResNet50HumanInference(ONNXInferenceManager):
	"""DeepLabV3+ ResNet50 Human Segmentation inference implementation."""
	
	def get_model_path(self):
		"""Return path to ResNet50 human segmentation model."""
		return os.path.join(project.folder, 'data', 'ml', 'models', 'resnet50-human', 'deeplabv3p-resnet50-human.onnx')
	
	def preprocess(self, nA):
		"""Preprocess input for ResNet50 human segmentation."""
		nA = self.npu.flip_v(nA)
		nA = self.npu.rgba_to_rgb(nA)
		nA = self.npu.grayscale_to_rgb(nA)
		
		# TouchDesigner textures are already 0-1 float, so denormalize to 0-255
		nA = self.npu.denormalize_td_image(nA)
		
		# PyTorch standardization: (x / 255 - mean) / std
		nA = self.npu.imagenet_normalize(nA)
		
		# This model expects [batch, height, width, channels] format (TensorFlow-style)
		# Just add batch dimension, keep HWC format
		input_tensor = np.expand_dims(nA, axis=0).astype('float32')
		
		return input_tensor
	
	def postprocess(self, outputs):
		"""Postprocess human segmentation output."""
		output = outputs[0]
		
		# Post-processing for segmentation mask
		# TensorFlow DeepLabV3 outputs [batch, height, width, classes]
		# Remove batch dimension
		if output.shape[0] == 1:
			output = output[0]  # Now [height, width, classes]
		
		# Get the predicted class for each pixel (argmax along last dimension)
		if len(output.shape) == 3:  # [height, width, classes]
			predicted_classes = np.argmax(output, axis=-1)  # [height, width]
		else:  # Already a single channel
			predicted_classes = output
		
		# Create color map for different body parts
		# 0: background (black), 1: unknown, 2: hair, 3: unknown, 4: glasses,
		# 5: top-clothes, 6-8: unknown, 9: bottom-clothes, 10: torso-skin,
		# 11-12: unknown, 13: face, 14-15: arms, 16-17: legs, 18-19: feet
		color_map = np.array([
			[0.0, 0.0, 0.0],      # 0: background - black
			[0.5, 0.5, 0.5],      # 1: unknown - gray
			[0.8, 0.3, 0.3],      # 2: hair - red
			[0.5, 0.5, 0.5],      # 3: unknown - gray
			[0.3, 0.8, 0.8],      # 4: glasses - cyan
			[0.3, 0.3, 0.8],      # 5: top-clothes - blue
			[0.5, 0.5, 0.5],      # 6: unknown - gray
			[0.5, 0.5, 0.5],      # 7: unknown - gray
			[0.5, 0.5, 0.5],      # 8: unknown - gray
			[0.8, 0.3, 0.8],      # 9: bottom-clothes - magenta
			[0.9, 0.7, 0.6],      # 10: torso-skin - skin tone
			[0.5, 0.5, 0.5],      # 11: unknown - gray
			[0.5, 0.5, 0.5],      # 12: unknown - gray
			[1.0, 0.8, 0.7],      # 13: face - light skin tone
			[0.3, 0.8, 0.3],      # 14: left-arm - green
			[0.3, 0.8, 0.3],      # 15: right-arm - green
			[0.8, 0.8, 0.3],      # 16: left-leg - yellow
			[0.8, 0.8, 0.3],      # 17: right-leg - yellow
			[0.8, 0.5, 0.3],      # 18: left-foot - orange
			[0.8, 0.5, 0.3],      # 19: right-foot - orange
		], dtype=np.float32)
		
		# Map predicted classes to colors
		output_img = color_map[predicted_classes]
		
		# Already 0-1 float for TouchDesigner
		output_img = self.npu.flip_v(output_img)
		
		return output_img


# Create global instance of the inference manager
inference_manager = ResNet50HumanInference()
inference_manager.opPerformance = op('constant_performance')


# TouchDesigner callback wrappers that delegate to the manager
def onSetupParameters(scriptOp):
	return inference_manager.onSetupParameters(scriptOp)


def onPulse(par):
	return inference_manager.onPulse(par)


def onCook(scriptOp):
	return inference_manager.onCook(scriptOp)
