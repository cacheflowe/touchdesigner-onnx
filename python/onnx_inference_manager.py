"""
ONNXInferenceManager - Base class for TouchDesigner ONNX model inference with threading

This class encapsulates all the common patterns for loading and running ONNX models
in TouchDesigner with threaded inference to avoid blocking the main render loop.

Usage:
    Create a subclass and implement:
    - get_model_path(): Return the path to your ONNX model
    - preprocess(nA): Transform input numpy array to model input tensor
    - postprocess(outputs): Transform model outputs to final numpy array
    
    Optional overrides:
    - get_session_options(): Customize ONNX session options
    - on_model_loaded(session): Called after model loads successfully
"""

import os
import threading
import numpy as np
import onnxruntime as ort
import math

# Import util modules (will be available in TouchDesigner context)
onnx_util = mod(f'{op.PyUtils}/onnx_util')
npu = mod(f'{op.PyUtils}/numpy_util')


class ONNXInferenceManager:
	"""Base class for managing ONNX model loading and threaded inference in TouchDesigner."""
	
	def __init__(self):
		# Threaded model-loading state
		self.loading_thread = None
		self.is_loading = False
		self.load_error = None
		
		# Threaded inference state
		self.inference_thread = None
		self.is_inferencing = False
		self.inference_lock = threading.Lock()
		self.pending_result = None  # Results from background thread
		self.input_tensor_cache = None  # Pre-processed input for thread
		self.frames_skipped = 0  # Track how many frames we've skipped
		self.frames_skipped_final = 0  # Final count of skipped frames to report
		
		# ONNX setup
		ort.preload_dlls(directory="")
		self.session = None  # ONNX session
		
		# Utils
		self.onnx_util = onnx_util
		self.npu = npu
	
	def printONNX(self, *args):
		"""Logging helper for ONNX operations."""
		print("[ONNX]", *args)
	
	# ========== Methods to Override in Subclasses ==========
	
	def get_model_path(self):
		"""
		Return the full path to the ONNX model file.
		Must be implemented by subclass.
		"""
		raise NotImplementedError("Subclass must implement get_model_path()")
	
	def preprocess(self, nA):
		"""
		Preprocess input numpy array to model input tensor.
		Must be implemented by subclass.
		
		Args:
			nA: Raw numpy array from TouchDesigner texture
		
		Returns:
			Preprocessed input tensor ready for model inference
		"""
		raise NotImplementedError("Subclass must implement preprocess()")
	
	def postprocess(self, outputs):
		"""
		Postprocess model outputs to final numpy array for TouchDesigner.
		Must be implemented by subclass.
		
		Args:
			outputs: Raw outputs from model.run()
		
		Returns:
			Final numpy array (H, W, C) in float32 format for TouchDesigner
		"""
		raise NotImplementedError("Subclass must implement postprocess()")
	
	def get_session_options(self):
		"""
		Override to customize ONNX session options.
		Returns None by default.
		"""
		return None
	
	def on_model_loaded(self, session):
		"""
		Called after model loads successfully.
		Override to perform additional setup.
		"""
		pass
	
	# ========== Model Loading ==========
	
	def loadONNX(self, scriptOp):
		"""Initiate threaded model loading."""
		if self.is_loading:
			self.printONNX("Model is already loading...")
			return
		
		# Reset session and start loading thread
		self.session = None
		scriptOp.par.Loadstatus = "loading"
		self.loading_thread = threading.Thread(target=self._load_model_thread)
		self.loading_thread.daemon = True
		self.loading_thread.start()
	
	def _load_model_thread(self):
		"""Background thread for loading ONNX model."""
		self.is_loading = True
		self.load_error = None
		
		try:
			self.printONNX('=============================================')
			self.printONNX("Starting ONNX model loading in background...")
			
			# Get model path from subclass
			model_path = self.get_model_path()
			self.printONNX("model:", model_path)
			
			# Get session options (if customized)
			sess_options = self.get_session_options()
			
			# Log ONNX environment
			if self.onnx_util:
				self.onnx_util.log_onnx_options()
				providers = self.onnx_util.providers()
			else:
				providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
			
			# Load model
			if sess_options:
				temp_session = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
			else:
				temp_session = ort.InferenceSession(model_path, providers=providers)
			
			self.printONNX('ONNX Device activated:', ort.get_device())
			self.printONNX('### session props -----------------------------------')
			
			if self.onnx_util:
				self.onnx_util.log_model_details(temp_session)
			
			# Call subclass hook
			self.on_model_loaded(temp_session)
			
			# Only assign to global session when fully loaded
			self.session = temp_session
			self.printONNX("ONNX model loaded successfully!")
			self.printONNX('=============================================')
			
		except Exception as e:
			self.load_error = str(e)
			self.printONNX(f"Error loading ONNX model: {e}")
		finally:
			self.is_loading = False
	
	def get_loading_status(self):
		"""Returns status of model loading."""
		if self.session is not None:
			return "loaded"
		elif self.is_loading:
			return "loading"
		elif self.load_error:
			return f"error: {self.load_error}"
		else:
			return "not_loaded"
	
	# ========== Threaded Inference ==========
	
	def _inference_thread(self):
		"""Background thread for preprocessing, ONNX inference, and post-processing."""
		try:
			# Call subclass preprocessing
			input_tensor = self.preprocess(self.input_tensor_cache)
			
			# Run inference
			outputs = self.session.run(None, {self.session.get_inputs()[0].name: input_tensor})
			
			# Call subclass postprocessing
			output_img = self.postprocess(outputs)
			
			# Ensure output is float32 for TouchDesigner
			output_img = output_img.astype(np.float32)
			
			# Store results thread-safely
			with self.inference_lock:
				self.pending_result = output_img
				
		except Exception as e:
			self.printONNX(f"Inference error: {e}")
			import traceback
			self.printONNX(traceback.format_exc())
		finally:
			self.is_inferencing = False
			self.frames_skipped_final = self.frames_skipped
	
	# ========== TouchDesigner Callbacks ==========
	
	def onSetupParameters(self, scriptOp):
		"""Setup custom parameters for the script operator."""
		page = scriptOp.appendCustomPage('Custom')
		# Add reload pulse
		page.appendPulse('Reloadonnx', label='Reload ONNX')
		# Add status info
		page.appendStr('Loadstatus', label='Load Status')
		scriptOp.par.Loadstatus = self.get_loading_status()
		return
	
	def onPulse(self, par):
		"""Handle custom pulse parameter triggers."""
		if par.name == 'Reloadonnx':
			self.session = None  # Reset the session
		return
	
	def onCook(self, scriptOp):
		"""
		Main inference loop called every frame by TouchDesigner.
		Handles model loading, result retrieval, and inference dispatching.
		"""
		# Update status parameter
		status = self.get_loading_status()
		scriptOp.par.Loadstatus = status
		
		# Make sure we've loaded the model
		if self.session is None:
			if not self.is_loading:
				self.loadONNX(scriptOp)
			# Return early if model isn't ready yet
			return
		
		# Check if we have a loading error
		if self.load_error:
			self.printONNX(f"Cannot process: {self.load_error}")
			return
		
		# Check if we have results from background thread
		with self.inference_lock:
			if self.pending_result is not None:
				output_img = self.pending_result
				self.pending_result = None
				self.frames_skipped = 0
				
				# Update performance metrics if available
				try:
					self.opPerformance.par.const0value = self.frames_skipped_final
					if self.frames_skipped_final > 0:  # Prevent div by zero
						self.opPerformance.par.const1value = math.floor(60 / self.frames_skipped_final)
				except:
					pass  # Performance constants not available
				
				# Output result directly (already fully processed)
				scriptOp.copyNumpyArray(output_img)
				return  # Early return after outputting result
		
		# If inference is still running, skip this frame (natural frame skipping via threading)
		if self.is_inferencing:
			self.frames_skipped += 1
			return
		
		# Capture input on main thread (GPU texture access only)
		try:
			inputTex = scriptOp.inputs[0]
			nA = inputTex.numpyArray(delayed=True)
			if nA is None:
				return
			
			# Store raw array for background thread to process
			# could do nA.copy() if worried about mutability
			self.input_tensor_cache = nA 
			
		except Exception as e:
			self.printONNX(f"Error capturing input: {e}")
			return
		
		# Start inference in background thread
		self.is_inferencing = True
		self.inference_thread = threading.Thread(target=self._inference_thread)
		self.inference_thread.daemon = True
		self.inference_thread.start()
		
		# Debug: Check thread count
		active_threads = threading.active_count()
		if active_threads > 10:
			self.printONNX(f"Warning: {active_threads} active threads!")
