import os
import numpy as np
import cv2

# Import the base inference manager
ONNXInferenceManager = mod(f'{op.PyUtils}/onnx_inference_manager').ONNXInferenceManager

class DepthAnythingInference(ONNXInferenceManager):
	"""Depth Anything / MiDaS depth estimation inference implementation."""
	
	def get_model_path(self):
		"""Return path to depth estimation model."""
		# midas v3
		# model_path = os.path.join(project.folder, 'data', 'ml', 'models', 'midas', 'dpt_beit_base_384.onnx')
		# model_path = os.path.join(project.folder, 'data', 'ml', 'models', 'midas', 'dpt_swin2_tiny_256.onnx') # good!
		# model_path = os.path.join(project.folder, 'data', 'ml', 'models', 'midas', 'midas_v21_small_256.onnx') # noisy, bad accuracy on detailed scenes
		# depthanything_v2
		model_path = os.path.join(project.folder, 'data', 'ml', 'models', 'depth-anything', 'depth_anything_v2_vits_dynamic.onnx')  # NICE - this is the best? 10ms
		# model_path = os.path.join(project.folder, 'data', 'ml', 'models', 'depth-anything', 'depth_anything_v2_vitb_indoor_dynamic.onnx') # Way more detail, 13ms, inverted output
		# model_path = os.path.join(project.folder, 'data', 'ml', 'models', 'depth-anything', 'depth_anything_v2_vitl_indoor_dynamic.onnx') # Slow but nice/detailed output
		# model_path = os.path.join(project.folder, 'data', 'ml', 'models', 'depth-anything-hf', 'model_fp16.onnx') # Nice but slower)
		# model_path = os.path.join(project.folder, 'data', 'ml', 'models', 'depth-anything-hf', 'model_q4f16.onnx') # Nice but slower - no real gains from quantization)
		return model_path
	
	def preprocess(self, nA):
		"""Preprocess input for depth estimation model."""
		nA = self.npu.flip_v(nA)
		nA = self.npu.rgba_to_rgb(nA)
		nA = self.npu.grayscale_to_rgb(nA)
		
		# Detect model type from session if available
		model_path = self.session._model_path if hasattr(self.session, '_model_path') else ""
		is_dpt_beit = "dpt_beit" in model_path.lower()
		newModel = "Midas-V2" not in model_path
		
		if newModel == False:
			nA = self.npu.denormalize_td_image(nA)

		# Special preprocessing
		if is_dpt_beit:
			if nA.shape[:2] != (384, 384):
				nA = cv2.resize(nA, (384, 384), interpolation=cv2.INTER_CUBIC)
		elif newModel == True:
			# ImageNet normalization
			nA = self.npu.imagenet_normalize(nA)

		# Prepare input tensor
		input_tensor = self.npu.add_batch_dimension(nA)
		input_tensor = self.npu.convert_to_float32(input_tensor)
		input_tensor = np.transpose(input_tensor, (0, 3, 1, 2))
		
		return input_tensor
	
	def postprocess(self, outputs):
		"""Postprocess depth estimation output."""
		depth_map = outputs[0]
		
		# Detect model type
		model_path = self.session._model_path if hasattr(self.session, '_model_path') else ""
		is_dpt_beit = "dpt_beit" in model_path.lower()
		
		# Post-processing (squeeze and normalize)
		output_img = np.squeeze(depth_map)
		
		if is_dpt_beit:
			if output_img.max() > output_img.min():
				output_img = (output_img - output_img.min()) / (output_img.max() - output_img.min()) * 255.0
			else:
				output_img = np.zeros_like(output_img)
			output_img = np.stack([output_img, output_img, output_img], axis=-1)
		else:
			output_img = (output_img - output_img.min()) / (output_img.max() - output_img.min()) * 255.0
			output_img = output_img.astype(np.uint8)
			output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
		
		# Final output formatting
		output_img = output_img.astype(np.float32) / 255.0
		output_img = self.npu.flip_v(output_img)
		
		return output_img


# Create global instance of the inference manager
inference_manager = DepthAnythingInference()
inference_manager.opPerformance = op('constant_performance')


# TouchDesigner callback wrappers that delegate to the manager
def onSetupParameters(scriptOp):
	return inference_manager.onSetupParameters(scriptOp)


def onPulse(par):
	return inference_manager.onPulse(par)


def onCook(scriptOp):
	return inference_manager.onCook(scriptOp)
