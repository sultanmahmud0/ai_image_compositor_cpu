import numpy as np
import cv2
from pathlib import Path
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image

def segment_object(image_path, model_path="models/sam_vit_h.pth", output_dir="outputs"):
    print("Segmenting object using SAM...")

    # Prepare paths
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load SAM model
    sam = sam_model_registry["vit_h"](checkpoint=model_path)
    sam.to(device="cpu")
    predictor = SamPredictor(sam)
    predictor.set_image(image_rgb)

    # Use a central point for initial segmentation
    h, w = image.shape[:2]
    input_point = np.array([[w // 2, h // 2]])
    input_label = np.array([1])

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    best_mask = masks[np.argmax(scores)]

    # Save binary mask
    mask_output_path = output_dir / f"{image_path.stem}_mask.png"
    mask_img = (best_mask * 255).astype(np.uint8)
    Image.fromarray(mask_img).save(mask_output_path)

    # Apply mask to RGB image
    masked_image = image_rgb.copy()
    masked_image[~best_mask] = 255  # white background

    segmented_output_path = output_dir / f"{image_path.stem}_segmented.png"
    Image.fromarray(masked_image).save(segmented_output_path)

    print(f"Saved segmented object to: {segmented_output_path}")
    print(f"Saved binary mask to: {mask_output_path}")

    return Image.fromarray(masked_image), Image.fromarray(mask_img)
