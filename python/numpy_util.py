import numpy
import cv2

def flip_v(nA):
    return cv2.flip(nA, 0)

def flip_h(nA):
    return cv2.flip(nA, 1)

def grayscale_to_rgb(nA):
    if len(nA.shape) == 2:  # If input is grayscale
        return cv2.cvtColor(nA, cv2.COLOR_GRAY2RGB)
    return nA

def rgba_to_rgb(nA):
    if nA.shape[2] == 4:  # If input is RGBA
        return nA[:, :, :3]  # Drop the alpha channel
    return nA

def convert_bgr_to_rgb(nA):
    return cv2.cvtColor(nA, cv2.COLOR_BGR2RGB)

def denormalize_td_image(nA):
    if nA.max() <= 1.0:
        return (nA * 255).astype(numpy.uint8)
    return nA

def ensure_dtype(nA, dtype=numpy.uint8):
    if nA.dtype != dtype:
        return nA.astype(dtype)
    return nA

def resize_image(nA, width, height):
    return cv2.resize(nA, (width, height))

def convert_to_float32(nA):
    return nA.astype(numpy.float32)

def convert_to_uint8(nA):
    return nA.astype(numpy.uint8)

def convert_to_int32(nA):
    return nA.astype(numpy.int32)

def add_batch_dimension(nA):
    return numpy.expand_dims(nA, axis=0)