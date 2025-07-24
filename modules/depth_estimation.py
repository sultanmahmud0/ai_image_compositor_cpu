# modules/depth_estimation.py

import torch
from transformers import DPTImageProcessor, DPTForDepthEstimation
from PIL import Image
import numpy as np
import os

def estimate_depth(input_image_path, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)

    # Load image
    image = Image.open(input_image_path).convert("RGB")

    # Load processor and model from the local folder with .safetensors
    processor = DPTImageProcessor.from_pretrained("models/dpt_large")
    model = DPTForDepthEstimation.from_pretrained("models/dpt_large")

    model.to("cpu")
    model.eval()

    # Prepare image for the model
    inputs = processor(images=image, return_tensors="pt").to("cpu")

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # Resize to the original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    # Normalize to 0–255 and convert to uint8
    depth = prediction.cpu().numpy()
    depth_min, depth_max = depth.min(), depth.max()
    depth_normalized = (255 * (depth - depth_min) / (depth_max - depth_min)).astype(np.uint8)

    # Save depth image
    output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(input_image_path))[0]}_depth.png")
    Image.fromarray(depth_normalized).save(output_path)

    print(f"✅ Saved depth map to: {output_path}")
    return depth_normalized
