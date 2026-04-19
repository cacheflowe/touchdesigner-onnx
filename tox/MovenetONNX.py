import os
import time

import threading
import numpy as np
import onnxruntime as ort

# custom util imports
onnx_util = mod(f'{op.PyUtils}/onnx_util')
npu = mod(f'{op.PyUtils}/numpy_util')


#####################################################
#####################################################
#####################################################
# Skeleton class for temporal tracking
#####################################################
#####################################################
#####################################################

class Skeleton:
	"""
	Skeleton object for temporal tracking.
	Optimized for performance with cached channel names and in-place updates.

	Temporal skeleton-tracking strategy ported from:
	https://github.com/cacheflowe/haxademic.js/blob/master/demo/demo-webcam-ml5-bodypose.js
	
	TODO:
	- How to ensure python libs with pyEnvManager are ready before component runs?
	- Catch errors and display in new COMP status field and maybe set the COMP color to red
	- Nice-to-haves:
		- Model path config in COMP parameters?
		- If only looking at torso, can we discard squished body parts?
	"""

	# Class variable for unique skeleton IDs
	_skeleton_count = 0
	
	# Keypoint names for MoveNet model
	KEYPOINT_NAMES = [
		"nose",
		"eye_l",
		"eye_r",
		"ear_l",
		"ear_r",
		"shoulder_l",
		"shoulder_r",
		"elbow_l",
		"elbow_r",
		"wrist_l",
		"wrist_r",
		"hip_l",
		"hip_r",
		"knee_l",
		"knee_r",
		"ankle_l",
		"ankle_r",
		"hand_l",  # Computed: extends from wrist
		"hand_r",  # Computed: extends from wrist
	]
	
	# Bounding box property names
	BBOX_PROPS = [
		"bbox_xmin",
		"bbox_ymin",
		"bbox_xmax",
		"bbox_ymax",
		"bbox_width",
		"bbox_height",
		"bbox_center_x",
		"bbox_center_y",
		"bbox_area",
		"confidence",
	]
	
	# Connections between keypoints for drawing skeleton
	CONNECTIONS = [
		(0, 1), (0, 2),  # nose to eyes
		(1, 3), (2, 4),  # eyes to ears
		(0, 5), (0, 6),  # nose to shoulders
		(5, 7), (7, 9),  # left arm
		(6, 8), (8, 10),  # right arm
		(5, 11), (6, 12),  # shoulders to hips
		(11, 13), (13, 15),  # left leg
		(12, 14), (14, 16)  # right leg
	]
	
	# Colors for visualization
	POINT_COLOR = (0, 255, 0)     # Green for keypoints
	LINE_COLOR = (255, 0, 0)      # Red for skeleton lines
	BOX_COLOR = (255, 255, 0)     # Yellow for bounding box
	POINT_SIZE = 2                # Point size for better visibility
	LINE_THICKNESS = 1            # Line thickness
	
	# Pre-compute keypoint indices for distance calculation (exclude extremities)
	DISTANCE_KEYPOINT_INDICES = [i for i, name in enumerate(KEYPOINT_NAMES) if not name.startswith('elbow') and not name.startswith('wrist') and not name.startswith('ankle') and not name.startswith('knee')]
	
	@classmethod
	def nextSkeletonId(cls):
		"""Generate next unique skeleton ID."""
		cls._skeleton_count += 1
		return cls._skeleton_count
	
	def __init__(self, chopIndex):
		self.index = chopIndex  # 1-based index for CHOP channel names
		
		# Pre-allocate numpy arrays for keypoints (19 keypoints x 2 coords: 17 from model + 2 computed hands)
		self.kp_x = np.zeros(19, dtype=np.float64)
		self.kp_y = np.zeros(19, dtype=np.float64)
		
		# Pre-allocate array for bounding box (10 values)
		self.bbox = np.zeros(10, dtype=np.float64)
		
		# Tracking metadata
		self.confidence = 0.0
		self.best_distance = 0.0
		self.birth = 0.0
		self.age = 0.0
		self.user_id = -1
		self.lost_frames = 0
		self.age_frames = 0
		
		# Pre-cache channel names (built once, reused every frame)
		# Store as interleaved pairs: [nose:tx, nose:ty, eye_l:tx, eye_l:ty, ...]
		self._chan_names_keypoints = []
		for name in Skeleton.KEYPOINT_NAMES:
			self._chan_names_keypoints.append(f"p{chopIndex}/{name}:tx")
			self._chan_names_keypoints.append(f"p{chopIndex}/{name}:ty")
		self._chan_names_bbox = [f"p{chopIndex}/{prop}" for prop in Skeleton.BBOX_PROPS]
		self._chan_name_user_id = f"p{chopIndex}/user_id"
		self._chan_name_birth = f"p{chopIndex}/birth"
		self._chan_name_age = f"p{chopIndex}/age"
		self._chan_name_best_distance = f"p{chopIndex}/best_distance"

	def resetData(self):
		"""Reset all keypoint and bounding box data to defaults."""
		self.kp_x.fill(0.0)
		self.kp_y.fill(0.0)
		self.bbox.fill(0.0)
		self.confidence = 0.0
		self.best_distance = 0.0

	def resetId(self):
		"""Reset tracking identity."""
		self.birth = 0.0
		self.user_id = -1
		self.lost_frames = 0
		self.age_frames = 0
		self.age = 0.0

	def copyData(self, skel2):
		"""Copy keypoint and bounding box data from another skeleton (in-place)."""
		np.copyto(self.kp_x, skel2.kp_x)
		np.copyto(self.kp_y, skel2.kp_y)
		np.copyto(self.bbox, skel2.bbox)
		self.confidence = skel2.confidence
		self.best_distance = skel2.best_distance
		self.birth = skel2.birth
		self.age = absTime.seconds - self.birth

	def copyUserId(self, skel2):
		"""Copy tracking identity from another skeleton."""
		self.user_id = skel2.user_id
		self.birth = skel2.birth
		self.age = absTime.seconds - self.birth

	def setDebug(self, distance):
		"""Store the match distance for debugging."""
		self.best_distance = distance

	def lerpFrom(self, skel2, amt):
		"""Lerp this skeleton's data toward another skeleton's data for smoothing."""
		inv_amt = 1.0 - amt
		# Vectorized lerp operations
		self.kp_x[:] = amt * self.kp_x + inv_amt * skel2.kp_x
		self.kp_y[:] = amt * self.kp_y + inv_amt * skel2.kp_y
		self.bbox[:] = amt * self.bbox + inv_amt * skel2.bbox

	def start(self):
		"""Initialize a new skeleton with unique ID and birth time."""
		self.birth = absTime.seconds
		self.age = 0.0
		self.age_frames = 0
		self.user_id = Skeleton.nextSkeletonId()
		return self.user_id

	def setFromKeypoints(self, person_data, minConfidence=0.1, minScale=0.1, maxScale=1.0):
		"""
		Populate skeleton data from MoveNet keypoints array.
		person_data: 56-element array for one person from MoveNet output
		"""
		# Extract bounding box (indices 51-54) and confidence (index 55)
		bbox_ymin = person_data[51]
		bbox_xmin = person_data[52]
		bbox_ymax = person_data[53]
		bbox_xmax = person_data[54]
		bbox_width = bbox_xmax - bbox_xmin
		bbox_height = bbox_ymax - bbox_ymin
		bbox_area = bbox_width * bbox_height
		person_score = person_data[55]

		# Check for bad data - filter by confidence and bounding box area
		# Area filtering helps reject both small skeletons and people too close to camera
		if person_score < minConfidence or bbox_area < minScale or bbox_area > maxScale:
			self.resetData()
			return

		# Set bounding box values directly into array
		self.bbox[0] = bbox_xmin
		self.bbox[1] = bbox_ymin
		self.bbox[2] = bbox_xmax
		self.bbox[3] = bbox_ymax
		self.bbox[4] = bbox_width
		self.bbox[5] = bbox_height
		self.bbox[6] = bbox_xmin + (bbox_width * 0.5)
		self.bbox[7] = bbox_ymin + (bbox_height * 0.5)
		self.bbox[8] = bbox_width * bbox_height
		self.bbox[9] = person_score  # confidence is last bbox prop
		self.confidence = person_score

		# Set keypoints (each has 3 values: y, x, score)
		for kp_idx in range(17):
			self.kp_x[kp_idx] = person_data[kp_idx * 3 + 1]
			self.kp_y[kp_idx] = 1.0 - person_data[kp_idx * 3]  # Flip Y for TD coordinates
		
		# Compute hand positions (extend past wrist based on elbow->wrist direction)
		# hand_l (index 17): elbow_l=7, wrist_l=9
		# hand_r (index 18): elbow_r=8, wrist_r=10
		extension_factor = 0.45  # Extend hand 40% of forearm length past wrist
		
		# Left hand
		dx_l = self.kp_x[9] - self.kp_x[7]  # wrist_l - elbow_l
		dy_l = self.kp_y[9] - self.kp_y[7]
		self.kp_x[17] = self.kp_x[9] + dx_l * extension_factor
		self.kp_y[17] = self.kp_y[9] + dy_l * extension_factor
		
		# Right hand
		dx_r = self.kp_x[10] - self.kp_x[8]  # wrist_r - elbow_r
		dy_r = self.kp_y[10] - self.kp_y[8]
		self.kp_x[18] = self.kp_x[10] + dx_r * extension_factor
		self.kp_y[18] = self.kp_y[10] + dy_r * extension_factor

	def keypointsDistance(self, skel2):
		"""
		Calculate weighted distance between keypoints of two skeletons.
		Uses pre-computed indices to skip extremities.
		"""
		total_distance = 0.0
		for idx in Skeleton.DISTANCE_KEYPOINT_INDICES:
			dx = self.kp_x[idx] - skel2.kp_x[idx]
			dy = self.kp_y[idx] - skel2.kp_y[idx]
			total_distance += (dx * dx + dy * dy) ** 0.5
		return total_distance

	def createChopChannels(self, outputOp):
		"""Create CHOP channels for this skeleton (called once)."""
		# Channels are already interleaved in _chan_names_keypoints
		for name in self._chan_names_keypoints:
			outputOp.appendChan(name)
		for name in self._chan_names_bbox:
			outputOp.appendChan(name)
		outputOp.appendChan(self._chan_name_user_id)
		outputOp.appendChan(self._chan_name_birth)
		outputOp.appendChan(self._chan_name_age)
		outputOp.appendChan(self._chan_name_best_distance)

	def updateChopValues(self, outputOp, min_age_frames=0, lost_frames_threshold=0):
		"""Update CHOP channel values (called every frame). Uses cached channel names.
		Skeletons younger than min_age_frames or lost beyond lost_frames_threshold
		output zeros so they vanish from downstream visualizations."""
		is_active = self.user_id != -1 and self.lost_frames <= lost_frames_threshold and self.age_frames >= min_age_frames

		# Update keypoints (interleaved: x, y, x, y, ...)
		for i in range(19):
			outputOp[self._chan_names_keypoints[i * 2]][0] = self.kp_x[i] if is_active else 0.0
			outputOp[self._chan_names_keypoints[i * 2 + 1]][0] = self.kp_y[i] if is_active else 0.0
		
		# Update bounding box
		for i in range(10):
			outputOp[self._chan_names_bbox[i]][0] = self.bbox[i] if is_active else 0.0
		
		# Update metadata
		outputOp[self._chan_name_user_id][0] = self.user_id if is_active else -1
		outputOp[self._chan_name_birth][0] = self.birth if is_active else 0.0
		outputOp[self._chan_name_age][0] = self.age if is_active else 0.0
		outputOp[self._chan_name_best_distance][0] = self.best_distance if is_active else 0.0

######################################################
######################################################
# MovenetONNX main class
######################################################
######################################################

class MovenetONNX:
	"""
	MoveNet ONNX wrapper for TouchDesigner.
	
	Provides real-time multi-person pose estimation using Google's MoveNet model
	with temporal skeleton tracking. Runs ONNX inference in a background thread
	for optimal performance.
	
	Features:
	- Threaded ONNX model loading and inference
	- Temporal skeleton tracking with unique user IDs
	- Smooth skeleton transitions via lerping
	- Lost skeleton recovery (re-matching after brief occlusions)
	- Optimized CHOP output with pre-cached channel names
	- Support for up to 6 simultaneous pose detections
	
	Output Data (per skeleton):
	- 17 keypoint positions (x, y coordinates)
	- Bounding box (min, max, width, height, center, area)
	- Tracking metadata (user_id, birth time, match distance, confidence)
	
	Usage:
	- Connect video input to null_input TOP
	- Call Update() each frame from a script DAT
	- Call OutputSkeletonsToChop() to populate the Script CHOP
	- Access skeleton data via output CHOP channels (or GetSkeletonsData())
	"""

	NUM_POSES = 6  # maximum number of poses to detect

	def __init__(self, ownerComp: baseCOMP):
		self.ownerComp: baseCOMP = ownerComp
		self.getOps()
		self.initONNX()
		self.initSkeletonTracking()

		# set defaults
		self.loggedInputShape = False
		self.ToggleDebug(self.ownerComp.par.Drawdebug.eval())
		self.ToggleDebug(self.ownerComp.par.Drawskeletononly.eval())
		self.queuedStatus = "None"
		self.queuedError = "None"
		
		# Performance timing variables
		self.time_read_top = 0.0
		self.time_copy_to_chop = 0.0
		self.time_inference = 0.0
		self.time_skeleton_tracking = 0.0
		self.time_model_load = 0.0
		self.num_active_skeletons = 0

	def getOps(self):
		# grab references to nodes
		self.opRawInputTOP: nullTOP = self.ownerComp.op('in1')
		self.opInputTOP: nullTOP = self.ownerComp.op('null_input')
		self.opConstantInputConfigCHOP: constantCHOP = self.ownerComp.op('constant_input_config')
		self.opScriptCHOP: scriptCHOP = self.ownerComp.op('script_movenet_chop')
		self.opOutputCHOP: scriptCHOP = self.ownerComp.op('constant1')
		self.opScriptTOP: scriptTOP = self.ownerComp.op('script_movenet_top')
		self.opInfoCHOP: infoCHOP = self.ownerComp.op('info_scripts')
		self.opSwitchInputTOP: switchTOP = self.ownerComp.op('switch_input_or_cv_debug')
		self.opPerformanceChop: constantCHOP = self.ownerComp.op('constant_performance')
		self.opSkeletonVisualizers: list[baseCOMP] = self.ownerComp.ops('Skeleton*')

	def InputW(self):
		# quantize to multiple of 32 for model compatibility
		return 32 * round(self.ownerComp.par.Inputwidth.eval() / 32)

	def InputH(self):
		# maintain aspect ratio based on input TOP
		# and quantize to multiple of 32 for model compatibility
		return 32 * round(self.ownerComp.par.Inputwidth.eval() * self.opRawInputTOP.height / self.opRawInputTOP.width / 32)

	def DebugW(self):
		return self.opRawInputTOP.width * self.ownerComp.par.Drawdebugscale.eval()

	def DebugH(self):
		return self.opRawInputTOP.height * self.ownerComp.par.Drawdebugscale.eval()

	def initONNX(self):
		self.session = None  
		self.loading_thread = None
		self.is_loading = False
		self.keypoints = None
		
		# Threaded inference state
		self.inference_thread = None
		self.is_inferencing = False
		self.inference_lock = threading.Lock()
		self.pending_keypoints = None  # Results from background thread
		self.input_tensor_cache = None  # Pre-processed input for thread
		self.frames_skipped = 0  # Track how many frames we've skipped

	def initSkeletonTracking(self):
		"""Initialize skeleton tracking arrays and parameters."""
		
		# Skeleton arrays — each slot is a persistent output position
		# Skeletons stay in their assigned slot until truly lost
		self.skeletons = []      # Persistent output slots (fixed 6)
		self.skeletonsNew = []   # Incoming frame data (temporary)
		
		# Create skeleton object pools
		for i in range(MovenetONNX.NUM_POSES):
			skelIndex = i + 1  # 1-based index for channel names
			self.skeletons.append(Skeleton(skelIndex))
			self.skeletonsNew.append(Skeleton(skelIndex))
		
		# Flag to track if CHOP channels have been built
		self.channelsBuilt = False

	def printONNX(self, *args):
		print("[MoveNet]", *args)

	def loadModel(self):
		# if self.is_loading:
		# 	self.printONNX("Model is already loading...")
		# 	return

		# Reset session and start loading thread
		self.session = None
		self.queuedStatus = "Loading model..."
		self.loading_thread = threading.Thread(target=self._loadModelThread)
		self.loading_thread.daemon = True
		self.loading_thread.start()		

	def setNodeGood(self):
		self.ownerComp.color = (0, 1, 0)

	def setNodeError(self):
		self.ownerComp.color = (1, 0, 0)

	def ToggleDebug(self, enable: bool):
		if enable == True:
			self.ownerComp.par.opviewer = './out_skeleton_debug'
			for viz in self.opSkeletonVisualizers:
				viz.allowCooking = True
		else:
			self.ownerComp.par.opviewer = './out_skeletons'
			for viz in self.opSkeletonVisualizers:
				viz.allowCooking = False

	def ToggleDrawSkeletonOnly(self, skeleton_only: bool):
		print(skeleton_only)
		if skeleton_only:
			for viz in self.opSkeletonVisualizers:
				viz.op('text_info').bypass = True
				viz.op('rectangle_bbox').bypass = True
		else:
			for viz in self.opSkeletonVisualizers:
				viz.op('text_info').bypass = False
				viz.op('rectangle_bbox').bypass = False


	# =================================================
	# Model loading and processing 
	# =================================================

	def lazyLoadModel(self):
		if self.session is None:
			if not self.is_loading:
				self.loadModel()

	def _loadModelThread(self):
		self.is_loading = True
		self.queuedStatus = "Loading..."

		try:
			start_time = time.perf_counter()
			self.printONNX('=============================================')
			self.printONNX("Starting ONNX model loading in background...")

			# Build paths & config
			model_path = os.path.join(project.folder, 'data', 'ml', 'models', 'movenet', 'movenet-multipose-lightning.onnx')
			self.printONNX("model:", model_path)

			# load model & provider
			onnx_util.log_onnx_options()
			providers = onnx_util.providers()
			temp_session = ort.InferenceSession(model_path, providers=providers)
			self.printONNX('ONNX Device activated:', ort.get_device())
			self.printONNX('### session props -----------------------------------')
			onnx_util.log_model_details(temp_session)
			# Only assign to global session when fully loaded
			self.session = temp_session
			self.time_model_load = time.perf_counter() - start_time
			self.printONNX(f"ONNX model loaded successfully in {self.time_model_load * 1000:.2f} ms!")
			self.printONNX('=============================================')
			self.queuedStatus = "Loaded"
			self.queuedError = None

		except Exception as e:
			self.queuedError = f"Error loading ONNX model: {str(e)}"
		finally:
			self.is_loading = False


	# =================================================
	# CHOP output helpers
	# =================================================

	# =================================================
	# Main update loop
	# =================================================

	def Update(self):
		self.lazyLoadModel()
		if self.session is not None:
			self.runInferenceThreaded()
			self.trackSkeletons()
		self.reportPerformanceStats()
		self.updateStatus()
		return

	def updateStatus(self):
		if self.queuedError:
			if self.queuedError != None:
				self.printONNX(self.queuedError)
				self.ownerComp.par.Errormessage = self.queuedError
				self.setNodeError()
		elif self.ownerComp.par.Errormessage != "None":
			self.ownerComp.par.Errormessage = "None"
		self.queuedError = None

		if self.queuedStatus:
			if self.queuedStatus != None:
				self.ownerComp.par.Modelloaded = self.queuedStatus
				self.setNodeGood()
			self.queuedStatus = None

	# =================================================
	# Skeleton temporal tracking
	# =================================================

	# Slot-stable skeleton tracking:
	# - Each of the 6 output slots is a persistent position in the CHOP.
	# - A skeleton assigned to slot N stays in slot N until truly lost — 
	#   other skeletons leaving/entering don't shift existing assignments.
	# - Matching uses greedy best-first on keypoint distance across all
	#   occupied slots and new detections simultaneously (prevents suboptimal
	#   per-row greedy matches).
	# - Lost skeletons stay in their slot with decaying confidence for
	#   recoveryFrames, enabling re-identification after brief occlusions.
	# - New skeletons fill the first empty slot (user_id == -1).
	# - Empty slots output all zeros with user_id == -1.

	def trackSkeletons(self):
		"""
		Slot-stable temporal skeleton tracking.
		Skeletons maintain their CHOP slot position across their lifetime.
		"""
		if not self.keypointsValid():
			return

		start_time = time.perf_counter()
		minConfidence = self.ownerComp.par.Minconfidence.eval()
		minScale = self.ownerComp.par.Minscale.eval()
		maxScale = self.ownerComp.par.Maxscale.eval()
		lerpAmp = self.ownerComp.par.Lerpamp.eval()
		maxMatchDist = self.ownerComp.par.Maxmatchdist.eval()
		recoveryFrames = self.ownerComp.par.Recoveryframes.eval()

		num_people = self.keypoints.shape[1]

		# Populate skeletonsNew from current keypoints
		for i in range(MovenetONNX.NUM_POSES):
			if i < num_people:
				self.skeletonsNew[i].setFromKeypoints(self.keypoints[0, i], minConfidence, minScale, maxScale)
			else:
				self.skeletonsNew[i].resetData()

		# Indices of valid new detections
		new_indices = [i for i in range(MovenetONNX.NUM_POSES) if self.skeletonsNew[i].confidence > 0.01]

		# Occupied slots: active or lost-but-still-recovering (user_id assigned)
		occupied_slots = [i for i in range(MovenetONNX.NUM_POSES) if self.skeletons[i].user_id != -1]

		# === Greedy best-first matching ===
		# Build all pairwise distances, sort by distance, assign greedily.
		# This avoids suboptimal per-row greedy matches when skeletons cross paths.
		matched_slots = set()
		matched_new = set()

		if occupied_slots and new_indices:
			pairs = []
			for slot_i in occupied_slots:
				for new_j in new_indices:
					dist = self.skeletons[slot_i].keypointsDistance(self.skeletonsNew[new_j])
					if dist < maxMatchDist:
						pairs.append((dist, slot_i, new_j))

			# Sort by distance (best matches first)
			pairs.sort(key=lambda p: p[0])

			# Greedy assignment: each slot and each detection matched at most once
			for dist, slot_i, new_j in pairs:
				if slot_i in matched_slots or new_j in matched_new:
					continue

				# Match found — lerp new data toward previous, preserve identity
				self.skeletonsNew[new_j].copyUserId(self.skeletons[slot_i])
				self.skeletonsNew[new_j].lerpFrom(self.skeletons[slot_i], lerpAmp)
				self.skeletonsNew[new_j].setDebug(dist)

				# Write lerped data back into the persistent slot
				self.skeletons[slot_i].copyData(self.skeletonsNew[new_j])
				self.skeletons[slot_i].copyUserId(self.skeletonsNew[new_j])
				self.skeletons[slot_i].lost_frames = 0
				self.skeletons[slot_i].age_frames += 1

				matched_slots.add(slot_i)
				matched_new.add(new_j)

		# === Handle unmatched occupied slots (lost skeletons) ===
		# Decay confidence and increment lost counter. Free slot when expired.
		for slot_i in occupied_slots:
			if slot_i in matched_slots:
				continue
			self.skeletons[slot_i].lost_frames += 1
			self.skeletons[slot_i].confidence *= 0.9  # Gradual fade
			if self.skeletons[slot_i].lost_frames > recoveryFrames:
				# Truly lost — free this slot for a new skeleton
				self.skeletons[slot_i].resetData()
				self.skeletons[slot_i].resetId()

		# === Assign unmatched new detections to empty slots ===
		# Before spawning a new skeleton, check if the detection is a duplicate
		# of an already-matched person. MoveNet can output multiple overlapping
		# detections for the same person — suppress these to avoid false spawns.
		empty_slots = [i for i in range(MovenetONNX.NUM_POSES) if self.skeletons[i].user_id == -1]
		for new_j in new_indices:
			if new_j in matched_new:
				continue
			# Duplicate suppression: if this detection is close to any
			# already-matched slot, it's a duplicate — discard it.
			is_duplicate = False
			for slot_i in matched_slots:
				dist = self.skeletons[slot_i].keypointsDistance(self.skeletonsNew[new_j])
				if dist < maxMatchDist:
					is_duplicate = True
					break
			if is_duplicate:
				continue
			if not empty_slots:
				break  # All 6 slots occupied
			slot_i = empty_slots.pop(0)
			self.skeletonsNew[new_j].start()  # Assign new unique user_id
			self.skeletons[slot_i].copyData(self.skeletonsNew[new_j])
			self.skeletons[slot_i].copyUserId(self.skeletonsNew[new_j])
			self.skeletons[slot_i].lost_frames = 0

		# Count active skeletons (any slot with an assigned user_id)
		self.num_active_skeletons = sum(1 for s in self.skeletons if s.user_id != -1)
		self.time_skeleton_tracking = time.perf_counter() - start_time

	def OutputSkeletonsToChop(self, opScriptCHOP: scriptCHOP):
		"""Write tracked skeleton data to the Script CHOP."""
		# Build channels once on first call
		if not self.channelsBuilt:
			self.buildChopChannels(opScriptCHOP)
			self.channelsBuilt = True
		
		# Update values each frame
		start_time = time.perf_counter()
		min_age = int(self.ownerComp.par.Minimumage.eval())
		lost_threshold = int(self.ownerComp.par.Lostframesthreshold.eval())
		for skel in self.skeletons:
			skel.updateChopValues(opScriptCHOP, min_age, lost_threshold)
		self.time_copy_to_chop = time.perf_counter() - start_time

	def buildChopChannels(self, outputOp):
		"""Build all CHOP channels once. Called on first frame only."""
		outputOp.clear()
		for skel in self.skeletons:
			skel.createChopChannels(outputOp)

	def RebuildChopChannels(self):
		"""Force rebuild of CHOP channels (call if structure changes)."""
		self.channelsBuilt = False

	def GetSkeletonsData(self):
		"""Get tracked skeleton data as a list of dictionaries."""
		skeletons_data = []
		for skel in self.skeletons:
			skel_data = {
				'kp_x': skel.kp_x.copy(),
				'kp_y': skel.kp_y.copy(),
				'bbox': skel.bbox.copy(),
				'confidence': skel.confidence,
				'user_id': skel.user_id,
				'birth': skel.birth,
				'age': skel.age,
				'best_distance': skel.best_distance
			}
			skeletons_data.append(skel_data)
		return skeletons_data
	
	# =================================================
	# Performance reporting
	# =================================================

	def reportPerformanceStats(self):
		self.opPerformanceChop.par.const0value = self.time_read_top * 1000
		self.opPerformanceChop.par.const1value = self.time_inference * 1000
		self.opPerformanceChop.par.const2value = self.time_skeleton_tracking * 1000
		self.opPerformanceChop.par.const3value = self.time_copy_to_chop * 1000
		self.opPerformanceChop.par.const4value = (self.time_read_top + self.time_inference + self.time_skeleton_tracking + self.time_copy_to_chop) * 1000
		self.opPerformanceChop.par.const5value = self.frames_skipped
		self.opPerformanceChop.par.const6value = self.num_active_skeletons
		self.opPerformanceChop.par.const7value = self.time_model_load * 1000


	# =================================================
	# Read and process input image
	# =================================================

	def runInferenceThreaded(self):
		"""
		Run MoveNet inference in a background thread.
		Skips starting new inference if previous one is still running.
		"""
		if self.session is None:
			return

		# Check if we have results from background thread
		with self.inference_lock:
			if self.pending_keypoints is not None:
				self.keypoints = self.pending_keypoints
				self.pending_keypoints = None
				self.frames_skipped = 0

		# If inference is still running, skip this frame
		if self.is_inferencing:
			self.frames_skipped += 1
			return

		# Capture and pre-process input on main thread (GPU texture access)
		try:
			start_time = time.perf_counter()
			nA = self.opInputTOP.numpyArray(delayed=True)
			if nA is None:
				return
			
			# Pre-process the numpy array
			nA = npu.flip_v(nA)
			nA = npu.rgba_to_rgb(nA)
			nA = npu.grayscale_to_rgb(nA)
			nA = npu.denormalize_td_image(nA)
			input_tensor = npu.add_batch_dimension(nA)
			self.input_tensor_cache = npu.convert_to_int32(input_tensor)
			self.time_read_top = time.perf_counter() - start_time
			
			# Log input shape once
			if not self.loggedInputShape:
				self.printONNX("Input TOP shape:", nA.shape)
				self.loggedInputShape = True
			
		except Exception as e:
			self.queuedError = f"Error capturing input: {e}"
			return

		# Start inference in background thread
		self.is_inferencing = True
		self.inference_thread = threading.Thread(target=self._inferenceThread)
		self.inference_thread.daemon = True
		self.inference_thread.start()

	def _inferenceThread(self):
		"""Background thread for ONNX inference."""
		try:
			start_time = time.perf_counter()
			input_name = self.session.get_inputs()[0].name
			output_name = self.session.get_outputs()[0].name
			result = self.session.run([output_name], {input_name: self.input_tensor_cache})[0]
			self.time_inference = time.perf_counter() - start_time
			
			# Store results thread-safely
			with self.inference_lock:
				self.pending_keypoints = result

			# should be good to reset error now
			self.queuedError = None
				
		except Exception as e:
			self.queuedError = f"Inference error: {e}"
		finally:
			self.is_inferencing = False

	def keypointsValid(self):
		"""Check if keypoints array has valid MoveNet format."""
		if self.keypoints is None:
			return False
		if self.keypoints.shape[0] == 1 and len(self.keypoints.shape) == 3 and self.keypoints.shape[2] == 56:
			return True
		return False












	# def onDestroyTD(self):
	# 	"""
	# 	Called when the extension or compeonent is being deleted. Use this
	# 	instead of __del__ for cleanup tasks.
	# 	"""
	# 	debug("onDestroyTD called")

	# def onInitTD(self):
	# 	"""
	# 	Called after the extension is fully initialized and attached to the 
	# 	component. Use this instead of __init__ for tasks that require other
	# 	components' extensions to be available, or that use promoted members.
	# 	"""
	# 	debug("onInitTD called")