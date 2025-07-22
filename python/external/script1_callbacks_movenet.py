import os
import sys

# Make sure our paths are set up before using external modules
print("🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍")


def addPath(new_path):
		if new_path not in sys.path:
				if os.path.exists(new_path):
						# Add to the beginning of the path list
						sys.path.insert(0, new_path)


utils_path = os.path.join(project.folder, 'python', 'util')
modules_path = os.path.join(project.folder, 'python', '_local_modules')

addPath(modules_path)
addPath(utils_path)

# print all paths
print('🐍 Python locations in sys.path:')
for path in sys.path:
		print("🐍 -", path)
print("🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍")

# import other dependencies now that the path supports it
import cv2
import importlib
import onnx_util  # our custom onnx utility module
import numpy_util as npu  # our custom numpy utility module
import onnxruntime as ort
import numpy as np
import threading

# reload our custom modules on save in case they've changed
importlib.reload(onnx_util)
importlib.reload(npu)

# Movenet debug globals -------------------------------

keypoint_names = [
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
]

bounding_box_props = [
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


# Define connections between keypoints for drawing skeleton
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

# Define colors for visualization
POINT_COLOR = (0, 255, 0)     # Green for keypoints
LINE_COLOR = (255, 0, 0)      # Red for skeleton lines
BOX_COLOR = (255, 255, 0)     # Yellow for bounding box
POINT_SIZE = 2                # Point size for better visibility
LINE_THICKNESS = 1            # Line thickness


# Threaded model-loading helpers -------------------------------

loading_thread = None
is_loading = False
load_error = None


# ONNX setup -------------------------------

session = None  # ONNX session
numOutputResults = 0  # Number of output results


def printONNX(*args):
		print("[ONNX]", *args)


def loadONNX(scriptOp):
		global session, loading_thread, is_loading

		if is_loading:
				printONNX("Model is already loading...")
				return

		# Reset session and start loading thread
		session = None
		scriptOp.par.Loadstatus = "loading"
		loading_thread = threading.Thread(target=_load_model_thread)
		loading_thread.daemon = True
		loading_thread.start()


def _load_model_thread():
		global session, is_loading, load_error

		is_loading = True
		load_error = None

		try:
				printONNX('=============================================')
				printONNX("Starting ONNX model loading in background...")

				# Build paths & config
				model_path = os.path.join(project.folder, 'data', 'ml', 'models', 'movenet', 'movenet-multipose-lightning.onnx')
				printONNX("model:", model_path)

				# load model & provider
				onnx_util.log_onnx_options()
				providers = onnx_util.providers()
				temp_session = ort.InferenceSession(model_path, providers=providers)
				printONNX('ONNX Device activated:', ort.get_device())
				printONNX('### session props -----------------------------------')
				onnx_util.log_model_details(temp_session)
				# Only assign to global session when fully loaded
				session = temp_session
				printONNX("ONNX model loaded successfully!")
				printONNX('=============================================')

		except Exception as e:
				load_error = str(e)
				printONNX(f"Error loading ONNX model: {e}")
		finally:
				is_loading = False


def get_loading_status():
		"""Returns status of model loading"""
		if session is not None:
				return "loaded"
		elif is_loading:
				return "loading"
		elif load_error:
				return f"error: {load_error}"
		else:
				return "not_loaded"

# CHOP output helpers -------------------------------


def chanName(index, name):
		return f"p{index+1}/{name}"


def chanNamePos(index, name, axis):
		return f"p{index+1}/{name}:{axis}"


def rebuildOutputChannels(scriptOp, maxResults):
		global numOutputResults
		# if maxResults is the same as the last time, do nothing
		if maxResults != numOutputResults:
				numOutputResults = maxResults
				scriptOp.clear()
		if scriptOp['p1/nose:tx'] is None:  # make sure we have a channel that we expect
				# print('Creating channels for skeleton detections')
				for i in range(maxResults):
						for name in keypoint_names:
								# create channel names
								addChannel(scriptOp, chanNamePos(i, name, 'tx'))
								addChannel(scriptOp, chanNamePos(i, name, 'ty'))
								addChannel(scriptOp, chanNamePos(i, name, 'tz'))
						# create bounding box channels
						for prop in bounding_box_props:
								addChannel(scriptOp, chanName(i, prop))


def addChannel(outputOp, chan_name):
		if outputOp[chan_name] is None:
				outputOp.appendChan(chan_name)
				# print(f"Created channel: {chan_name}")
		else:
				# print(f"Channel already exists: {chan_name}")
				pass


def setValue(outputOp, index, name, value):
		# print('Setting value:', name, value)
		chan_name = chanName(index, name)
		outputOp[chan_name][0] = value
		return


def setValuePos(outputOp, index, name, x, y, z):
		# set the x, y, z values for a keypoint
		setValue(outputOp, index, f"{name}:tx", x)
		setValue(outputOp, index, f"{name}:ty", y)
		setValue(outputOp, index, f"{name}:tz", z)
		return


def resetDetectionResult(outputOp, index):
		# reset detection values to 0
		for name in keypoint_names:
				setValue(outputOp, index, name, 0.0)
		return


# Shared Script Op callbacks -------------------------------

# press 'Setup Parameters' in the OP to call this function to re-create the parameters.
def onSetupParameters(scriptOp):
		page = scriptOp.appendCustomPage('Custom')

		# add debug toggle and default to 1
		page.appendStr('Topinput', label='TOP Input')
		page.appendPulse('Reloadonnx', label='Reload ONNX')
		# add status info
		page.appendStr('Loadstatus', label='Load Status')
		scriptOp.par.Loadstatus = get_loading_status()
		page.appendToggle('Drawdebug', label='Draw Debug')
		scriptOp.par.Drawdebug = 1
		page.appendInt('Maxresults', label='Maximum Results')
		scriptOp.par.Maxresults = 6
		page.appendFloat('Minconfidence', label='Minimum Confidence Threshold')
		scriptOp.par.Minconfidence = 0.1
		page.appendFloat('Minscale', label='Minimum bbox scale')
		scriptOp.par.Minscale = 0.1
		page.appendToggle('Normalized', label='Normalized Output')
		scriptOp.par.Normalized = 1
		return


# called whenever custom pulse parameter is pushed
def onPulse(par):
		global session
		if par.name == 'Reloadonnx':
				session = None  # reset the session
		return

def onCook(scriptOp):
		global session, is_loading, load_error
		
		# Check TOP vs CHOP to allow use in either type of script op
		scriptType = scriptOp.family
		isTOP = scriptType == 'TOP'
		isCHOP = scriptType == 'CHOP'

		# get custom params
		inputPath = scriptOp.par.Topinput.eval()
		drawDebug = scriptOp.par.Drawdebug.eval()
		minConfidence = scriptOp.par.Minconfidence.eval() if scriptOp else 0.1
		maxResults = scriptOp.par.Maxresults.eval() if scriptOp else 6

		# get input image and force cook
		# inputTex = scriptOp.inputs[0] 
		inputTex = op(inputPath)
		nA = inputTex.numpyArray(delayed=False)

		# Update status parameter
		status = get_loading_status()
		scriptOp.par.Loadstatus = status

		# make sure we've loaded the model
		if session is None:
				if not is_loading:
						if isCHOP:
								scriptOp.clear()
						loadONNX(scriptOp)

				# Return early if model isn't ready yet
				return

		# Check if we have a loading error
		if load_error:
				printONNX(f"Cannot process: {load_error}")
				return

		# pre-process the numpy array, ensuring it is in the correct format for mediapipe
		nA = npu.flip_v(nA)
		nA = npu.rgba_to_rgb(nA)
		nA = npu.grayscale_to_rgb(nA)
		nA = npu.denormalize_td_image(nA)
		input_tensor = npu.add_batch_dimension(nA)
		input_tensor = npu.convert_to_int32(input_tensor) # Convert to int32 for Movenet

		# Run inference
		input_name = session.get_inputs()[0].name
		output_name = session.get_outputs()[0].name
		keypoints = session.run([output_name], {input_name: input_tensor})[0]
	
		# output numeric data to CHOP
		if isCHOP:
			rebuildOutputChannels(scriptOp, maxResults)
			store_data(scriptOp, keypoints)

		# Draw the detection results on the image
		if isTOP:
				if drawDebug:
						annotated_image = visualize_poses(scriptOp, nA, keypoints)
						annotated_image = cv2.flip(annotated_image, 0)
						scriptOp.copyNumpyArray(annotated_image)
				# draw numpy image array back to output
				nA = npu.flip_v(nA)
				scriptOp.copyNumpyArray(nA)

		return


def keypoints_valid(keypoints):
		"""Check if keypoints are valid"""
		if keypoints.shape[0] == 1 and len(keypoints.shape) == 3 and keypoints.shape[2] == 56:
				return True
		return False


def store_data(scriptOp, keypoints):
		"""Store bounding boxes, confidence scores, and keypoints in the DAT tables"""
		# TODO: set data as default arrays if it's low confidence or low scale
		minConfidence = scriptOp.par.Minconfidence.eval() if scriptOp else 0.1
		minScale = scriptOp.par.Minscale.eval() if scriptOp else 0.1

		if keypoints_valid(keypoints):
				num_people = keypoints.shape[1]  # Up to 6 people
				for person_idx in range(num_people):
						person_data = keypoints[0, person_idx]

						# Extract bounding box coordinates (index 51-54)
						bbox_ymin = person_data[51]
						bbox_xmin = person_data[52]
						bbox_ymax = person_data[53]
						bbox_xmax = person_data[54]
						bbox_width = bbox_xmax - bbox_xmin
						bbox_height = bbox_ymax - bbox_ymin
						bbox_center_x = bbox_xmin + (bbox_width / 2)
						bbox_center_y = bbox_ymin + (bbox_height / 2)
						bbox_area = bbox_width * bbox_height

						# Extract confidence score (index 55)
						person_score = person_data[55]

						# check bad data
						badData = person_score < minConfidence or bbox_height < minScale
						if badData:
								bbox_xmin = 0.0
								bbox_ymin = 0.0
								bbox_xmax = 0.0
								bbox_ymax = 0.0
								bbox_center_x = 0.0
								bbox_center_y = 0.0
								bbox_area = 0.0
								bbox_width = 0.0
								bbox_height = 0.0
								person_score = 0.0

						# Set bounding box properties
						setValue(scriptOp, person_idx, 'bbox_xmin', bbox_xmin)
						setValue(scriptOp, person_idx, 'bbox_ymin', bbox_ymin)
						setValue(scriptOp, person_idx, 'bbox_xmax', bbox_xmax)
						setValue(scriptOp, person_idx, 'bbox_ymax', bbox_ymax)
						setValue(scriptOp, person_idx, 'bbox_width', bbox_width)
						setValue(scriptOp, person_idx, 'bbox_height', bbox_height)
						setValue(scriptOp, person_idx, 'bbox_center_x', bbox_center_x)
						setValue(scriptOp, person_idx, 'bbox_center_y', bbox_center_y)
						setValue(scriptOp, person_idx, 'bbox_area', bbox_area)
						setValue(scriptOp, person_idx, 'confidence', person_score)

						# Skip low-confidence, low-scale detections
						if not badData:

								# Collect all keypoints for the person
								keypoint_data = [f"person_{person_idx}"]
								for kp_idx in range(17):
										x = float(person_data[kp_idx * 3 + 1])
										# flip data to match TD coordinates
										y = 1 - float(person_data[kp_idx * 3])
										z = 0.0  # add z for compatibility with 3D keypoints
										keypoint_data.extend([x, y, z])
										setValuePos(scriptOp, person_idx,
																keypoint_names[kp_idx], x, y, z)

						else:
								for kp_idx in range(17):
										# Reset keypoint values to 0
										setValuePos(scriptOp, person_idx,
																keypoint_names[kp_idx], 0.0, 0.0, 0.0)


def visualize_poses(scriptOp, frame, keypoints):
		"""Draw pose keypoints and connections on the frame"""
		minConfidence = scriptOp.par.Minconfidence.eval() if scriptOp else 0.1
		minScale = scriptOp.par.Minscale.eval() if scriptOp else 0.1
		h, w = frame.shape[:2]  # Get frame dimensions

		# Process and visualize keypoints based on their format
		if keypoints.shape[0] == 1 and len(keypoints.shape) == 3 and keypoints.shape[2] == 56:
				# Flattened format: [batch, person_id, data]
				# Up to 6 people, but there will always be 6 people, with different confidence scores
				num_people = keypoints.shape[1]
				for person_idx in range(num_people):
						person_data = keypoints[0, person_idx]

						# Extract person's confidence score (index 55)
						person_score = person_data[55]
						# and draw bounding box
						bbox_ymin = person_data[51]
						bbox_xmin = person_data[52]
						bbox_ymax = person_data[53]
						bbox_xmax = person_data[54]
						bbox_h = bbox_ymax - bbox_ymin

						# print(f"Person {person_idx + 1} confidence: {person_score:.2f} minConfidence: {minConfidence}")
						# Skip low-confidence detections
						if person_score < minConfidence or bbox_h < minScale:
								continue

						points = []

						# Process all 17 keypoints
						for kp_idx in range(17):
								# Each keypoint has 3 values [y, x, score] starting at index 0
								y = float(person_data[kp_idx * 3])
								x = float(person_data[kp_idx * 3 + 1])
								score = float(person_data[kp_idx * 3 + 2])

								# Convert normalized coordinates to pixel coordinates
								x_px = int(x * w)
								y_px = int(y * h)
								points.append((x_px, y_px, score))

								cv2.circle(frame, (x_px, y_px), POINT_SIZE, POINT_COLOR, -1)

								# Label keypoint (optional)
								# cv2.putText(frame, f"{keypoint_names[kp_idx]}",
								# 						(x_px + 5, y_px - 5), cv2.FONT_HERSHEY_SIMPLEX,
								# 						0.5, POINT_COLOR, 1)

						# Draw skeleton connections
						for connection in CONNECTIONS:
								start_idx, end_idx = connection
								if len(points) > max(start_idx, end_idx) and points[start_idx][2] > 0.2 and points[end_idx][2] > 0.2:
										start_point = (points[start_idx][0], points[start_idx][1])
										end_point = (points[end_idx][0], points[end_idx][1])
										cv2.line(frame, start_point, end_point,
														LINE_COLOR, LINE_THICKNESS)

						# Convert normalized coordinates to pixel coordinates
						x1 = int(bbox_xmin * w)
						y1 = int(bbox_ymin * h)
						x2 = int(bbox_xmax * w)
						y2 = int(bbox_ymax * h)

						# Draw bounding box
						cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 1)

		return frame
