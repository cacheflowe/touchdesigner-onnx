# ONNX in TouchDesigner

ONNX (Open Neural Network Exchange) is a highly portable ML model format that is (relatively) easily used across different frameworks and platforms. There are some nice computer vision (cv) models that allow for various creative opportunities in TouchDesigner. This repository has examples for **DepthAnythingV2** and **Movenet**.

In the journey to get these models working, I learned a lot about ONNX models, but even more about how external Python modules work in TouchDesigner. This README explains how to get the project running, but also serves as a reference to some less-documented ways that external Python tools can be used in TD.

## Installation


To install the required Python modules, use the shell scripts in the `python/scripts` directory. These scripts will automatically use TouchDesigner's built-in Python executable to install the necessary packages. Read more about how this works (and why it might not work) below.

1. Open a terminal or command prompt.
2. `cd` to the `python/scripts` directory.
2. Run the appropriate script for your operating system:
   - Windows: `install-modules.cmd` (or double-click the file)
   - Mac: `sudo install-modules.sh`
4. Download the models as described below.
5. Open `onnx-example.toe` in TouchDesigner 2023.12370

Notes:

- If you're on a Windows PC, it's expected that you have an NVIDIA GPU. Otherwise, you can switch to the CPU version of `onnxruntime` by editing the `requirements.txt` file to use `onnxruntime` instead of `onnxruntime-gpu`, and then re-running the install script. The CPU version is slower, but should work on non-NVIDIA GPUs.
- If you need to reinstall the Python modules, you can delete the `python/_local_modules` directory and re-run the install script. This will ensure that the modules are installed fresh. You'll need to quit TouchDesigner before deleting & reinstalling.

### The ONNX models

#### DepthAnythingV2

This is a model that performs depth estimation from a single image, and is useful for creating 3D effects in TouchDesigner. It's like having a Kinect or Orbbec depth map, but from any image or RGB webcam input. The model included in this example it the "small" (vits) version, which runs faster (but has lower output quality) than the larger versions. A good rule of thumb is to use the smallest model that gives you the results you want, because larger models are generally slower and more resource-intensive.

> This model was [downloaded](https://github.com/fabio-sim/Depth-Anything-ONNX/releases/download/v2.0.0/depth_anything_v2_vits_dynamic.onnx) from the [GitHub repository](https://github.com/fabio-sim/Depth-Anything-ONNX/releases), and moved into `data/ml/models/depth-anything/`

#### Movenet

This is a model that performs human pose estimation, and is useful for tracking body movements in TouchDesigner. The model included in this example is the "multipose" version, which can track multiple (up to 6) people in the same frame. At the time of this writing, this model handles multiple skeletons far better than MediaPipe's pose estimation.

> This model was [downloaded](https://huggingface.co/Xenova/movenet-multipose-lightning/resolve/main/onnx/model.onnx?download=true) from the [HuggingFace repository](https://huggingface.co/Xenova/movenet-multipose-lightning/tree/main/onnx), and renamed from `model.onnx` to `movenet_multipose_lightning.onnx` and moved into `data/ml/models/movenet/`

## Python Approach

General Python setup in this project:

1) We need to install external Python modules to run ONNX models in TouchDesigner. These are installed in a relatively unconventional way (using TouchDesigner's own Python executable), but this method has some advantages and dangers, noted below.

2) Since we're already relying on pip-installed modules, we'll import some of our own external Python code for common helper functions. These tools are located in `python/util` and make it easier to work with ONNX models in TouchDesigner. 

3) Finally, the Script nodes' Python code has been externalized (located in `python/external`) so it's easier to update and reference on GitHub. There's a bit of duplicated code between the **Movenet** and **DepthAnything** scripts, but this allows them to work independently and load their models on a Python thread so TD doesn't lock up quite as much. 

### Python modules installation 

General concepts:

- The install shell scripts use TouchDesigner's **built-in Python executable** to run `pip`, so everything that we install is compatible. This is a different approach than using a Python package manager like Conda, which is a commonly-accepted (and safe) way to manage external Python modules. By using TouchDesigner's instance of Python, we don't create a secondary Python executable or entire environment, and we can guarantee that all installed modules will be compiled against TouchDesigner's Python version. However, we risk polluting other Python environments.
- When the required Python modules are installed with `pip`, they're installed in a directory *inside of our project*, rather than in a virtual environment, or default system `pip` modules location. This is because the OS-specific install scripts specify `--target="../_local_modules"` in the `pip` command. When these scripts finish, there will be a directory of Python modules in the `python/_local_modules` directory - you can ignore these files. This directory has been added to .gitignore, because like `node_modules` in JavaScript, this folder can be very large and should not be committed to our project's repository. Anybody running the project should install these modules on their own machine.

### Default TouchDesigner Versions

This project works in TouchDesigner 2023.12370 and probably slightly earlier versions. It will almost certainly not work in later versions, as Cuda and numpy versions have changed in the latest experimental release.

TD 2023.12370 was built with these relevant tooling versions:

- `Python 3.11.1` (noted when the Textport opens)
- `Cuda 11.8` (noted in the [TD docs](https://derivative.ca/UserGuide/CUDA))
- `numpy 1.24.1` (check the version with `import numpy; print(numpy.__version__)` in the Textport)

### What is installed with the shell scripts

The shell scripts install the following Python modules from `requirements.txt`:

```
numpy==1.26.4
onnxruntime-gpu==1.18.0 # or onnxruntime==1.18.0 on Mac
```

- The `onnxruntime` (or `onnxruntime-gpu` on Windows) package loads and runs ONNX models. This package relies on a specific version of `numpy`, which is why it's in `requirements.txt`.
- TouchDesigner has its own version of `numpy` but it's not compatible with the version of `onnxruntime` that's compatible with this version of TouchDesigner 🫠, so we need to install a compatible version. The version of `numpy` we install is `1.26.4`, which is the last version before `numpy 2.x`, which could break other things in this version of TouchDesigner. The currently-[experimental versions of TD](https://derivative.ca/download/experimental) (2025.30060+) have `numpy` 2.x.
- `pip` can be helpful when accidentally installing incompatible versions with errors like: `onnxruntime-gpu 1.18.0 depends on numpy>=1.24.2`

### Windows-specific notes

On Windows, ONNX models will be GPU-accelerated with the `onnxruntime-gpu` package (version 1.18), which is compatible with Cuda 11.8, which is built into this version of TouchDesigner. Using the Textport, it's possible to confirm that the `onnxruntime-gpu` package can use the GPU by running the following commands in Textport:

```python
import onnxruntime as ort
print(ort.get_available_providers())
```

This will print a list of available providers, which should include `CUDAExecutionProvider` if the GPU is available and properly configured.

- The shell script installs the Python ONNX Runtime, which allows us to load and run any ONNX model in TouchDesigner. Each ONNX model has its own format and challenges to understand how to use it, but I've provided a few examples to get you started. This version of TouchDesigner needs version 1.18, which *should* support GPU inference on Windows, because it's numpy & Cuda versions are compatible.
  - `onnxruntime-gpu` on Windows
  - `onnxruntime` on Mac 

## Using your own Python modules in TouchDesigner

- As noted, this project has some local, custom Python modules that I've written to help with ONNX models. These modules are located in the `python/util` directory, and are imported into the main script nodes in the project.
- To make these local modules available to TouchDesigner, there needs to be an empty `__init__.py` file in the same directory to be recognized as a package. 
- Much like adding the path to the `pip`-installed packages, we need to add our local package path to `sys.path` in our scripts (noted below).
- This can be a really nice way to build your own reusable Python tools!

## Notes on Python package management in TD

- `conda`, `venv`, and `uv` are popular Python environment managers, but can make things more complex for a TD setup
  - By default, these package managers create isolated environments with *their own version of Python*. When creating a new environment, you need to ensure that **the virtual environment's Python version matches the one used by TouchDesigner** (3.11 in this project). This is easy to overlook, and can lead to any number of compatibility issues if the versions don't match. This is because Python modules are built to work with a specific version of Python, and if the versions don't match, you can get potentially-mysterious errors when your code runs.
- As an alternate approach, we can use TouchDesigner's own versions of Python and `pip` to install packages without needing to create a new environment.
  - TD's Python can be located on Windows & Mac at the following default locations (which might be slightly different on your system, depending on your installation):
    - Windows: `C:\Program Files\Derivative\TouchDesigner\bin\python.exe`
    - Mac: `/Applications/TouchDesigner.app/Contents/Frameworks/Python.framework/Versions/3.11/bin/python3.11` 
      - On Mac, right-click on the TouchDesigner app in Finder, select "Show Package Contents", then navigate to find the Python executable
- TouchDesigner already has its own versions of required tools like `numpy`, and regardless of which tool you use to install Python dependencies, you need to ensure that the versions of these tools match the ones used by TouchDesigner, or are close enough to not have breaking API changes between the versions, for risk of breaking other TD functionality. In this project, we're installing a specific version of `numpy` that is compatible with both TouchDesigner and ONNX Runtime. This kind of Python dependency version management is where things get tricky, and might not end up working the way you hope!

## Notes on the TD Python path

- TD looks in multiple directories for python modules. We add our custom modules paths to the front of that list, so that when TD finds a version of a tool that we're using, it uses that first (`numpy` for example). This can be tricky to navigate
- When TouchDesigner launches, before any additional Python paths are added, we can see the default paths by running the following code, or a one-liner int the Textport:

```python
for path in sys.path:
    print("-", path)
```

```python
print('\n'.join(f"- {path}" for path in sys.path))
```

The result looks like this:

```
- C:\Program Files\Derivative\TouchDesigner\bin
- C:\Program Files\Derivative\TouchDesigner\bin\python311.zip
- C:\Program Files\Derivative\TouchDesigner\bin\Lib
- C:\Program Files\Derivative\TouchDesigner\bin\DLLs
- C:\Program Files\Derivative\TouchDesigner\bin
- C:\Program Files\Derivative\TouchDesigner\bin\Lib\site-packages
```

This shows the default paths that TouchDesigner uses to look for Python modules. When we install our own modules, we add their location(s) to the front of this list, so that they take precedence over any other versions of the same module that might be installed elsewhere on the system. This is done by the following python code in in the Script nodes:

```python
import sys
import os

def addPath(new_path):
	if new_path not in sys.path: # Check if the path is already in the list (to avoid duplicates)
		if os.path.exists(new_path): # Make sure the path exists on disk
			sys.path.insert(0, new_path)  # Add to the beginning of the path list
	else:
		print('Python path already loaded!')
```

One tricky thing that could come up if you start installing python packages with TouchDesigner's pip is that it will install them into the system Python cache, which is not what we want. This is because TD's Python environment is not writable. If you run `pip install`, you will likely see an error like: `Defaulting to user installation because normal site-packages is not writeable`, which then means the packages are installed in a user-specific location, tucked away on your computer in an odd location, like this:

```
C:\Users\username\AppData\Roaming\Python\Python311\site-packages
```

If you've done this, without specifying a location with `--target`, your TouchDesigner path will have automatically added this odd location! The output of the above code would now show something like this:

```
- C:\Program Files\Derivative\TouchDesigner\bin
- C:\Program Files\Derivative\TouchDesigner\bin\python311.zip
- C:\Program Files\Derivative\TouchDesigner\bin\Lib
- C:\Program Files\Derivative\TouchDesigner\bin\DLLs
- C:\Users\username\AppData\Roaming\Python\Python311\site-packages # note this new path!
- C:\Program Files\Derivative\TouchDesigner\bin
- C:\Program Files\Derivative\TouchDesigner\bin\Lib\site-packages
```

To remove this path and undo any unwanted pip installation that you may have done, you can delete the `Python311` directory in the `AppData\Roaming\Python` directory and restart TouchDesigner. The extra path should be gone now if you log `sys.path` again. However, this could break other Python work on your computer if you've relied on this Python user path for other projects! So this is a potentially messy situation, and why environment management with a tool like Conda is generally a great idea. But if we're careful about using `--target` when installing Python modules in this project (as is done with the shell scripts), we should avoid this issue altogether.

You can always find the version and *system location* of a python module with code like this in the Textport. This could alert you to a version mismatch or unexpected location of a module that you thought was installed in your project:

```python
import sys; print('Python version:', sys.version); import numpy; print('Numpy version:', numpy.__version__); print('Numpy location:', numpy.__file__)
```

If the module is the TD default, you'd find it here:
- `C:\Program Files\Derivative\TouchDesigner\bin\Lib\site-packages\numpy\__init__.py`

But if it was installed in the user-specific location, you'd find it here:
- `C:\Users\cacheflowe\AppData\Roaming\Python\Python311\site-packages\numpy\__init__.py`

One final note on this: if you already have a global instance of Python 3.11 and you've used `pip` to install packages before getting this project running, the methodology in this project may not work as expected, especially if you've installed conflicting versions of the required packages. This is a tradeoff of "simplicity" vs adding more tooling that can help with environment isolation. Phew!

## ONNX Notes

- Performance and image inputs
  - ONNX models have optimal input sizes (often square formats, like 256x256 or 518x518), and in some cases might not accept anything outside of specific dimensions. The ONNX "input shape" is defined in the model file itself, and you can find it by using a tool like Netron, or by printing it to the Textport after loading the model into Python. In this example code, the input and output shapes are printed via a helper function in `onnx_util.py`. These model-preferred image sizes are generally the size that the model was trained at, and using different sizes may lead to unexpected results, errors, or slower processing.
  - In this example project, both models (DepthAnythingV2 & Movenet) accept *dynamic input sizes*, and there is a Fit TOP that scales the input to keep the aspect ratio (based on width), which can be set in the custom COMP params. It's recommended to keep this as low as you can, or performance can start to really degrade. However, if you want higher-fidelity results from the model and you don't need real-time performance, you can increase the Input Width. You might want to turn off TD's Realtime setting if you want to process a video frame-by-frame. Fortunately, the models only run when the input TOP cooks, so if you have a static image, you can safely run at a higher resolution after the first cook.
  - Using ML models in real-time is generally slow, which is why we scale down our images as they're used for inference. The models are often trained on large images, but we can use smaller images for real-time applications. Downscaling is a common strategy to speed up computationally-expensive graphical operations!
- Dynamic input sizes
  - Despite the dynamic input size, odd dimensions can create problems for the model. It's recommended to stay within power-of-two dimensions (256x256, 512x512, etc), or at least make sure the dimensions are evenly divisible by 16 or 32, which are more likely to succeed. 
  - Movenet has been slightly more finicky about the input size, and there's a section below about potential errors.
- ONNX models
  - ONNX models can be tricky to work with, especially if you're not familiar with the model's input and output shapes and numeric formats. It's important to understand the model's requirements and how to preprocess the input data before passing it to the model. I've tried implementing different models, and some work better than others. 
  - You might find an ONNX model that has different versions that are quantized or optimized, and these aren't guaranteed to be a drop-in replacement for the models that are working in this example.
  - [Netron](https://netron.app/) is a web-app that can visualize an ONNX model. This is useful for understanding the model's input and output shapes, as well as the layers and operations used in the model. But this requires ML knowledge to understand on a deeper level.
  - There are some additional helpful notes about running ONNX with Python in [this article](https://medium.com/data-science/how-to-run-stable-diffusion-with-onnx-dafd2d29cd14).
- TD script nodes, inference, and debugging
  - In the Movenet example, we're sharing a single script for both the TOP and CHOP script nodes. For a model like Movenet, we ultimately want numeric skeletal joint data, but we could also benefit from using `cv2` to draw the debug results to make sure the data is correct! So in the example, we can toggle between the TOP/CHOP script nodes, since we really only want one running at a time for performance. Inside the shared script, there are checks for whether it's running in a TOP or CHOP context to toggle any code that is exclusive to one or the other
  - This is where `numpy` comes into play. `numpy` is a Python library that is used across ML frameworks to handle large arrays of data. In this project it's used to transform an input image into numeric data that can be processed by an ONNX model, and then to transform the output data back into an image format that can be used in TouchDesigner. This is relevant to the ONNX model *input shape* and *output shape*, which are defined by the ONNX model itself. The input shape is the size and format of the image that the model expects, and the output shape is the size and format of the data that the model returns. In some cases, the input data needs to be transformed in a very specific way for a given model, such as normalizing the pixel values or changing the color space or batch size. The output data may also need to be transformed, like converting the output coordinates or colors back to normalized values.
    - Because of these common transformations, I've added a `numpy_util.py` and `onnx_util.py` module to the `python/util` directory, to reduce duplicated code when dealing with ONNX models.
    - Full disclosure: I used an AI coding assistant to help me through these steps, and I still don't feel like an expert here.


## Movenet-specific errors

If you see an error like this when running the Movenet example:

```
AddV2: right operand cannot broadcast on dim 2 LeftShape: {1,64,16,16}, RightShape: {1,64,15,16}
```

Something about the dimensions of the input isn't compatible with the model. The dimensions need to be within certain thresholds of a common divisor.

If you can't easily change the input TOP size to match the model's requirements, you can use a second Fit TOP in the COMP that doesn't maintain the aspect ratio. Instead, it letterboxes the input within a square input size. The adaptive Fit TOP is an ideal default, but might not always work.

Here's and example of an optional Python solution, but I prefer to keep resizing in TOPs, and out of the scripts:

```python
def get_model_friendly_size(width, height, base=32):
    # Round dimensions to multiples of base
    new_width = ((width + base - 1) // base) * base
    new_height = ((height + base - 1) // base) * base
    return new_width, new_height

def process_frame(frame):
    # Get current frame dimensions
    h, w = frame.shape[:2]
    
    # Round to nearest multiple of 32 for MoveNet compatibility
    target_w, target_h = get_model_friendly_size(w, h, 32)
    
    # Resize if needed
    if w != target_w or h != target_h:
        frame = cv2.resize(frame, (target_w, target_h))
        print(f"Resized from {w}x{h} to {target_w}x{target_h}")
```


## Research and resources used to create this project:

- Native onnx is supported in TD but only if you're building a custom node w/C++: 
  - https://github.com/TouchDesigner/CustomOperatorSamples/tree/main/TOP/ONNXCandyStyleTOP
- Otherwise, you can use the `onnxruntime-gpu` python package to run onnx models in TD.   
- ONNX info:
  - https://onnxruntime.ai/docs/install/
  - https://pypi.org/project/onnxruntime/
    - Version 1.18 is GPU compatible!
- Examples of ML tools and Python module management in TD
  - https://derivative.ca/community-post/tutorial/anaconda-miniconda-managing-python-environments-and-3rd-party-libraries
  - https://derivative.ca/community-post/real-time-magic-integrating-touchdesigner-and-onnx-models/69856
  - https://github.com/olegchomp/TDDepthAnything - A different method of running DepthAnything in TD
  - https://github.com/DBraun/PyTorchTOP
  - https://forum.derivative.ca/t/import-pytorch-torch-in-build-2021-39010/245984/18
  👈 Check the comments
  - https://github.com/IntentDev/TopArray/
  - https://github.com/ioannismihailidis/venvBuilderTD/
- Find more ONNX models: 
	- https://aihub.qualcomm.com/models
  - https://huggingface.co/qualcomm
  - https://huggingface.co/onnx-community/models
  - https://huggingface.co/onnxmodelzoo
  - https://docs.ultralytics.com/integrations/onnx/
- More documentation about using Python in TD
  - https://derivative.ca/UserGuide/Category:Python
  - https://derivative.ca/community-post/tutorial/external-python-libraries/61022
  - [How to use external python libraries in Touchdesigner](https://www.youtube.com/watch?v=_U5gcTEsupE)
  - [TouchDesigner | External Python Libraries | 7/8](https://www.youtube.com/watch?v=LFWcsx2Ic6g)