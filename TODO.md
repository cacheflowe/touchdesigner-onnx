# TODO

- This repo serves a few purposes:
  - Showing how to use ONNX models in TD. There's a world of ONNX models out there, and this is a good starting point
  - With this combination of models, we're at a point where ML models can replicate the features of a specialty camera like the Kinect. It's not 1-to-1, but it's pretty close, with the depth and skeletal tracking.
- This is a good time to point to Torin's [link] alternate method of using DepthAnythingV2 in TD, which was released at the same time as this example. Some pros for each version:
  - Torin's tools are probably the better choice in most cases!
    - Runs faster on Macs
    - Works across versions of TD (this ONNX implementation will need to be updated for the POPs version of TD)
    - Much easier to set up
    - Combined with Torin's MediaPipe plugin provides a similar level of "Kinect replacement" tooling
  - ONNX version
    - The main advantage to use any tools in this example is that Movenet runs super fast and can easily detect up to 6 skeletons, where Mediapipe struggles with more than a single skeleton
    - The models run at any resolution (not just square) - not sure if this is a limitation w/Torins!
    - Some of these tools might run faster on an NVIDIA GPU (needs to be tested)


- Add switches for experimental version
  - rename to 2023/2025?
    - onnxruntime-gpu[cuda,cudnn]==1.22.0
    - needs this line of python if you use the pip method: `ort.preload_dlls(directory="")`
  - externalize toxes to share
  - in script python, just remove the local modules path
- Add notes about Torin's version, pros/cons, and kinect replacement idea
  - Kinect reference for onnx library 
  - And note about pops & experimental version
  - Make link to torins libs and also note the differences
- Add toggle for Fit square/adaptive for both models
- Default to Movenet only turned on, since it performs better
- Add temporal tracking to movenet skeletons, with user IDs

- Download best models in shell scripts
- Can the shell scripts run automatically?



- TODO: build a shell script to do this automatically
  - Movenet (multipose):
    - https://huggingface.co/Xenova/movenet-multipose-lightning/tree/main/onnx
      - https://huggingface.co/Xenova/movenet-multipose-lightning/resolve/main/onnx/model.onnx?download=true (rename `model.onnx` to `movenet_multipose_lightning.onnx`)
      - Try 16-bit version, but code probably has to change
  - DepthAnythingV2:
    - https://github.com/fabio-sim/Depth-Anything-ONNX/releases
      - https://github.com/fabio-sim/Depth-Anything-ONNX/releases/download/v2.0.0/depth_anything_v2_vits_dynamic.onnx
    - Check this: https://huggingface.co/garryling/depth_anything_v2_optimized
    - https://huggingface.co/onnx-community/depth-anything-v2-small/tree/main/onnx
