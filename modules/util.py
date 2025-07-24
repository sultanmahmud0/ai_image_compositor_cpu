
from PIL import Image
from pathlib import Path

def get_rgb_image_from_path(image_path):
    image_path = Path(image_path)
    image_rgb = Image.open(image_path).convert("RGB")
    return image_rgb