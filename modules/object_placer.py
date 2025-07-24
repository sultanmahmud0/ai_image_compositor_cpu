# modules/object_placer.py
from typing import Any

import numpy as np
from PIL import Image
from numpy import floating


def get_average_depth(depth_map: np.ndarray, bbox: tuple[int, int, int, int]) -> float | floating[Any]:
    x1, y1, x2, y2 = bbox
    region = depth_map[y1:y2, x1:x2]
    if region.size == 0:
        return 1.0
    return np.mean(region)

def calculate_scale(foreground_size: tuple[int, int], background_depth: float, ref_depth: float = 1.0) -> tuple[int, int]:
    """
    Scale object inversely with depth. Make sure foreground_size is a tuple.
    """
    if not isinstance(foreground_size, tuple):
        raise TypeError(f"Expected tuple for size, got {type(foreground_size)}: {foreground_size}")

    w, h = foreground_size
    scale = ref_depth / background_depth
    scale = np.clip(scale, 0.4, 2.0)  # Prevent too tiny or too big
    return int(w * scale), int(h * scale)

def resize_object(object_image: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    return object_image.resize(target_size, Image.Resampling.LANCZOS)

def blend_object(background: Image.Image, object_image: Image.Image, mask: Image.Image, position: tuple[int, int]) -> Image.Image:
    """
    Blend object using mask at the given position (top-left corner).
    """
    background = background.convert("RGBA")
    object_image = object_image.convert("RGBA")
    mask = mask.convert("L")

    # Resize mask to match object
    mask = mask.resize(object_image.size)

    # Create a blank image for placement
    composed = background.copy()
    x, y = position
    composed.paste(object_image, (x, y), mask=mask)
    return composed
