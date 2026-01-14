import os
import numpy as np
import cv2

# Import the base inference manager
ONNXInferenceManager = mod(f'{op.PyUtils}/onnx_inference_manager').ONNXInferenceManager


class ModNetInference(ONNXInferenceManager):
	"""MODNet (Matting Objective Decomposition Network) portrait matting inference implementation."""
	
	def get_model_path(self):
		"""Return path to ModNet model."""
		return os.path.join(project.folder, 'data', 'ml', 'models', 'modnet-webnn', 'model.onnx')
	
	def preprocess(self, nA):
		"""Preprocess input for ModNet model."""
		nA = self.npu.flip_v(nA)
		nA = self.npu.rgba_to_rgb(nA)
		nA = self.npu.grayscale_to_rgb(nA)
		
		# TouchDesigner textures are already 0-1 float, so denormalize to 0-255
		nA = self.npu.denormalize_td_image(nA)
		
		# MODNet normalization: (x - 127.5) / 127.5 -> scales to -1 to 1
		nA = (nA.astype('float32') - 127.5) / 127.5
		
		# Transpose to CHW format and add batch dimension
		input_tensor = nA.transpose(2, 0, 1)
		input_tensor = np.expand_dims(input_tensor, axis=0).astype('float32')
		
		return input_tensor
	
	def postprocess(self, outputs):
		"""Postprocess ModNet alpha matte output."""
		# ModNet outputs an alpha matte (single channel, 0-1 range)
		alpha_matte = outputs[0]
		
		# Remove batch dimension and get first channel if needed
		alpha_matte = np.squeeze(alpha_matte)
		
		# Ensure it's in 0-1 range (ModNet should already output this)
		alpha_matte = np.clip(alpha_matte, 0.0, 1.0)
		
		# Convert to RGB by stacking the alpha channel 3 times
		# (or could use green channel for better visibility)
		# output_img = np.stack([alpha_matte * 0.0, alpha_matte, alpha_matte * 0.0], axis=-1)  # Green alpha
		output_img = np.stack([alpha_matte, alpha_matte, alpha_matte], axis=-1) # Alternative grayscale: 
		
		# Already in 0-1 float range for TouchDesigner
		output_img = self.npu.flip_v(output_img)
		
		return output_img


# Create global instance of the inference manager
inference_manager = ModNetInference()
inference_manager.opPerformance = op('constant_performance')

# TouchDesigner callback wrappers that delegate to the manager
def onSetupParameters(scriptOp):
	return inference_manager.onSetupParameters(scriptOp)


def onPulse(par):
	return inference_manager.onPulse(par)


def onCook(scriptOp):
	return inference_manager.onCook(scriptOp)
