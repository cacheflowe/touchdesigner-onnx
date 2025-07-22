import os
import sys

# Make sure our paths are set up before using external modules
print("🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍")

def addPath(new_path):
	if new_path not in sys.path:
		if os.path.exists(new_path):
			sys.path.insert(0, new_path)  # Add to the beginning of the path list

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
import threading
import numpy as np
import onnxruntime as ort
import numpy_util as npu  # our custom numpy utility module
import onnx_util  # our custom onnx utility module
import importlib
import cv2

# reload our custom modules on save in case they've changed
importlib.reload(onnx_util)
importlib.reload(npu)

# Threaded model-loading helpers -------------------------------

loading_thread = None
is_loading = False
load_error = None

# ONNX setup -------------------------------

session = None  # ONNX session

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
        # midas v3
        model_path = os.path.join(project.folder, 'data', 'ml', 'models', 'midas', 'dpt_beit_base_384.onnx')
        # model_path = os.path.join(project.folder, 'data', 'ml', 'models', 'midas', 'dpt_swin2_tiny_256.onnx') # good!
        # model_path = os.path.join(project.folder, 'data', 'ml', 'models', 'midas', 'midas_v21_small_256.onnx') # noisy, bad accuracy on detailed scenes
        # depthanything_v2
        model_path = os.path.join(project.folder, 'data', 'ml', 'models', 'depth-anything', 'depth_anything_v2_vits_dynamic.onnx')  # NICE - this is the best? 10ms
        # model_path = os.path.join(project.folder, 'data', 'ml', 'models', 'depth-anything', 'depth_anything_v2_vitb_indoor_dynamic.onnx') # Way more detail, 13ms, inverted output
        # model_path = os.path.join(project.folder, 'data', 'ml', 'models', 'depth-anything', 'depth_anything_v2_vitl_indoor_dynamic.onnx') # Slow but nice/detailed output
        # model_path = os.path.join(project.folder, 'data', 'ml', 'models', 'depth-anything-hf', 'model_fp16.onnx') # Nice but slower)
        # model_path = os.path.join(project.folder, 'data', 'ml', 'models', 'depth-anything-hf', 'model_q4f16.onnx') # Nice but slower - no real gains from quantization)
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


# Shared Script Op callbacks -------------------------------

# press 'Setup Parameters' in the OP to call this function to re-create the parameters.
def onSetupParameters(scriptOp):
    page = scriptOp.appendCustomPage('Custom')
    # add reload pulse
    page.appendPulse('Reloadonnx', label='Reload ONNX')
    # add status info
    page.appendStr('Loadstatus', label='Load Status')
    scriptOp.par.Loadstatus = get_loading_status()
    return


# called whenever custom pulse parameter is pushed
def onPulse(par):
    if par.name == 'Reloadonnx':
        session = None  # reset the session
    return


def onCook(scriptOp):
    global session, is_loading, load_error

    # Update status parameter
    status = get_loading_status()
    scriptOp.par.Loadstatus = status

    # make sure we've loaded the model
    if session is None:
        if not is_loading:
            loadONNX(scriptOp)
        # Return early if model isn't ready yet
        return

    # Check if we have a loading error
    if load_error:
        printONNX(f"Cannot process: {load_error}")
        return

    # Detect model-specific needs
    model_path = session._model_path if hasattr(session, '_model_path') else ""
    is_dpt_beit = "dpt_beit" in model_path.lower()
    newModel = "Midas-V2" not in model_path # all Midas after v2 handle normalized values

    # get input image
    inputTex = scriptOp.inputs[0]
    nA = inputTex.numpyArray(delayed=False)
    # Store original dimensions for later resizing
    orig_h, orig_w = nA.shape[:2]

    # pre-process the numpy array, ensuring it is in the correct format for mediapipe
    nA = npu.flip_v(nA)
    nA = npu.rgba_to_rgb(nA)
    nA = npu.grayscale_to_rgb(nA)
    if newModel == False:
        nA = npu.denormalize_td_image(nA)

    # special preprocessing
    if is_dpt_beit:
        # Special handling for DPT-BEiT model - force exact 384x384
        if nA.shape[:2] != (384, 384):
            nA = cv2.resize(nA, (384, 384), interpolation=cv2.INTER_CUBIC)
    elif newModel == True:
        # ImageNet normalization values
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        nA = (nA - mean) / std  # Apply ImageNet normalization

    # Ensure the input tensor is in the correct shape and type for the ONNX model
    input_tensor = npu.add_batch_dimension(nA)
    input_tensor = npu.convert_to_float32(input_tensor)
    # Transpose from (B, H, W, C) to (B, C, H, W) for the ONNX model
    input_tensor = np.transpose(input_tensor, (0, 3, 1, 2))

    # Run inference
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    depth_map = session.run([output_name], {input_name: input_tensor})[0]

    # squeeze: remove the batch dimension
    output_img = np.squeeze(depth_map)

    # Handle output based on model type
    if is_dpt_beit:
        # Normalize for display
        if output_img.max() > output_img.min():
            output_img = (output_img - output_img.min()) / (output_img.max() - output_img.min()) * 255.0
        else:
            output_img = np.zeros_like(output_img)

        # Convert to grayscale RGB for TouchDesigner
        output_img = np.stack([output_img, output_img, output_img], axis=-1)
    else:
        # For other models, normalize 
        output_img = (output_img - output_img.min()) / (output_img.max() - output_img.min()) * 255.0
        output_img = output_img.astype(np.uint8)

        # Convert from BGR (OpenCV default) to RGB for TouchDesigner
        output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)

    # output result
    # Convert to float32 and normalize to 0-1 range for TOP output
    output_img = output_img.astype(np.float32) / 255.0
    # Flip vertically to match TouchDesigner's coordinate system
    output_img = npu.flip_v(output_img)
    scriptOp.copyNumpyArray(output_img)
