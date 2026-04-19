# Threading Optimization: MoveNet Pattern for ONNX Inference

## Summary

Restructuring where preprocess/postprocess run relative to the background inference thread yielded a **massive performance gain** — from ~15-20 FPS to **70+ FPS** — with no changes to the actual inference or processing logic.

## The Problem (Before)

The original `ONNXInferenceManager` base class spawned a background thread that did **everything**:

```
Main Thread (onCook):
  - nA.copy()  ← copy TD's GPU staging buffer (required because bg thread can't read it safely)
  - Spawn thread

Background Thread:
  - preprocess(nA_copy)   ← 25-97ms (cache-cold numpy array from GPU staging buffer copy)
  - session.run()          ← ~16ms (CUDA inference)
  - postprocess(outputs)   ← 0.3-2ms
  - [thread holds is_inferencing=True for total duration: 40-115ms+]

Main Thread (next frames):
  - Skips frames while bg thread is busy → low effective FPS
```

**Key issues:**
1. `nA.copy()` was needed because the background thread couldn't safely read TD's GPU staging buffer
2. The copy itself was slow, and the bg thread reading the copied array still had cache misses
3. The thread held `is_inferencing=True` for the entire pre+infer+post duration, causing many skipped frames
4. Postprocess on the bg thread required thread-safety locks (e.g., `tracker_lock` in YOLO26)

## The Discovery

MoveNet's standalone implementation (`tox/MovenetONNX.py`) used a different pattern and ran significantly faster. The key difference: **only `session.run()` runs on the background thread**.

## The Solution (After)

Restructured `ONNXInferenceManager.onCook()` and `_inference_thread()` to match MoveNet's pattern:

```
Main Thread (onCook):
  1. Check for pending raw outputs from previous frame's bg thread
  2. If found: postprocess(raw_outputs) → copyNumpyArray → output
  3. If not inferencing: preprocess(nA) → store input tensor → spawn bg thread

Background Thread:
  - session.run() ONLY  ← ~16ms
  - Store raw outputs in pending_result

Main Thread (next onCook):
  - Pick up raw outputs, postprocess, output
```

## Why This Is So Much Faster

### 1. No `.copy()` needed
Preprocess now runs on the main thread, reading directly from TD's GPU staging buffer while it's cache-warm. The background thread never touches the raw TD buffer.

### 2. Background thread occupies minimum time
Previously: `is_inferencing=True` for 40-115ms+ (pre+infer+post)
Now: `is_inferencing=True` for ~16ms (infer only)

This means far fewer frames are skipped between inferences. The main thread can start a new inference almost every other frame instead of every 4-7 frames.

### 3. No thread-safety locks needed
Since postprocess runs on the main thread, there's no concurrent access to tracking state, table DATs, or other TD operators. Locks like `tracker_lock` in YOLO26 were removed entirely.

### 4. TD operator access is safe
Postprocess can directly write to Table DATs, update parameters, etc. without queuing or deferred updates.

## Files Changed

- **`python/onnx_inference_manager.py`** — Base class restructured:
  - `_inference_thread()`: Now only runs `session.run()`, stores raw outputs in `self.pending_result`
  - `onCook()`: Preprocess before thread spawn, postprocess when results arrive, removed `nA.copy()`
  
- **`python/script1_callbacks_yolo26_obj_det.py`** — Removed thread-safety overhead:
  - Removed `import threading`
  - Removed `self.tracker_lock = threading.Lock()`
  - Removed `with self.tracker_lock:` from `postprocess()` and `write_tracks_to_table()`

## Results

| Metric | Before | After |
|--------|--------|-------|
| Effective FPS | ~15-20 | **70+** |
| BG thread busy time | 40-115ms | ~16ms |
| Frames skipped per inference | 4-7 | 0-1 |
| Thread-safety locks needed | Yes | No |
| nA.copy() required | Yes | No |

## Applying to Other Scripts

This pattern should be applied to any ONNX inference script in the project. Scripts that already use `ONNXInferenceManager` as a base class (like Depth Anything) automatically benefit from the base class change. Standalone scripts need manual restructuring:

- **`script1_callbacks_yunet.py`** — Standalone (uses cv2.FaceDetectorYN, not ONNX Runtime). Could benefit from the same pattern if restructured to use a bg thread for `detector.detect()`.
- **`tox/MovenetONNX.py`** — Already uses this pattern (it was the inspiration). No changes needed.
- **Any future ONNX scripts** — Should extend `ONNXInferenceManager` to get this pattern for free.

## Rule of Thumb

> **Only `session.run()` (or equivalent blocking inference call) should run on the background thread.** Everything else — reading input textures, preprocessing numpy arrays, postprocessing outputs, writing to TD operators — belongs on the main thread.
