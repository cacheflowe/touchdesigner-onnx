import os
import numpy as np
import cv2

# Import the base inference manager
ONNXInferenceManager = mod(f'{op.PyUtils}/onnx_inference_manager').ONNXInferenceManager

class DepthAnythingInference(ONNXInferenceManager):
	"""Depth Anything / MiDaS depth estimation inference implementation."""

	def __init__(self):
		super().__init__()
		self._input_tensor_buf = None
		self._input_buf_shape = None
		self._output_buf = None
		self._output_buf_shape = None
		self._model_type = None  # 'dpt_beit', 'midas_v2', or 'modern'
		# Pre-computed ImageNet scale/offset: pixel * scale + offset
		# Equivalent to (pixel - mean) / std but avoids per-element division
		self._inet_scale = np.array([1.0/0.229, 1.0/0.224, 1.0/0.225], dtype=np.float32)
		self._inet_offset = np.array([-0.485/0.229, -0.456/0.224, -0.406/0.225], dtype=np.float32)

	def on_model_loaded(self, session):
		"""Detect model type once at load time instead of every frame."""
		model_path = self.get_model_path().lower()
		if "dpt_beit" in model_path:
			self._model_type = 'dpt_beit'
		elif "midas-v2" in model_path or "midas_v2" in model_path:
			self._model_type = 'midas_v2'
		else:
			self._model_type = 'modern'
		self.printONNX(f"Depth model type: {self._model_type}")
	
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
		"""Preprocess input for depth estimation model.
		Input nA is float32 RGBA 0-1 from TouchDesigner (bottom-up).
		"""
		h, w = nA.shape[:2]
		num_ch = nA.shape[2] if len(nA.shape) == 3 else 1

		if self._model_type == 'dpt_beit':
			# dpt_beit expects 0-255 float32, resized to 384x384
			# Flip + RGBA->RGB + denormalize in one pass
			if num_ch >= 3:
				rgb = nA[::-1, :, :3] * 255.0  # flip view + slice + scale
			else:
				rgb = np.stack([nA[::-1]] * 3, axis=-1) * 255.0
			if rgb.shape[:2] != (384, 384):
				rgb = cv2.resize(rgb, (384, 384), interpolation=cv2.INTER_CUBIC)
			return np.ascontiguousarray(rgb.transpose(2, 0, 1)[np.newaxis], dtype=np.float32)

		elif self._model_type == 'midas_v2':
			# MiDaS v2 expects 0-255 float32, no ImageNet normalization
			if num_ch >= 3:
				rgb = nA[::-1, :, :3] * 255.0
			else:
				rgb = np.stack([nA[::-1]] * 3, axis=-1) * 255.0
			return np.ascontiguousarray(rgb.transpose(2, 0, 1)[np.newaxis], dtype=np.float32)

		else:
			# Modern models (Depth Anything): ImageNet normalize from 0-1 directly
			# (x - mean) / std, where mean/std are for 0-1 range
			needed = (1, 3, h, w)
			if self._input_buf_shape != needed:
				self._input_tensor_buf = np.empty(needed, dtype=np.float32)
				self._input_buf_shape = needed
			# Flip + RGB channel copy into pre-allocated NCHW buffer
			if num_ch >= 3:
				flipped = nA[::-1, :, :3]  # view, no alloc
			else:
				flipped = np.stack([nA[::-1]] * 3, axis=-1)
			# ImageNet normalize: pixel * scale + offset (no per-element division)
			self._input_tensor_buf[0, 0] = flipped[:, :, 0] * self._inet_scale[0] + self._inet_offset[0]
			self._input_tensor_buf[0, 1] = flipped[:, :, 1] * self._inet_scale[1] + self._inet_offset[1]
			self._input_tensor_buf[0, 2] = flipped[:, :, 2] * self._inet_scale[2] + self._inet_offset[2]
			return self._input_tensor_buf
	
	def postprocess(self, outputs):
		"""Postprocess depth estimation output.
		Normalizes depth map to 0-1 float32 RGB for TouchDesigner.
		"""
		depth = np.squeeze(outputs[0])  # (H, W)
		h, w = depth.shape[:2]

		# Normalize to 0-1 in-place (no temp array allocation)
		d_min = depth.min()
		d_range = depth.max() - d_min
		if d_range > 0:
			np.subtract(depth, d_min, out=depth)
			np.multiply(depth, 1.0 / d_range, out=depth)
		else:
			depth[:] = 0

		# Expand grayscale to RGB + flip via single broadcast assignment
		needed = (h, w, 3)
		if self._output_buf is None or self._output_buf_shape != needed:
			self._output_buf = np.empty(needed, dtype=np.float32)
			self._output_buf_shape = needed
		self._output_buf[:] = depth[::-1, :, np.newaxis]  # flip + broadcast (H,W,1)->(H,W,3)

		return self._output_buf


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
