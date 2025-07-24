import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import cv2

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = clip_model.to(device)

def get_mask_regions(mask, label_id):
    contours, _ = cv2.findContours(
        (mask == label_id).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    bboxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 30 and h > 30:  # filter tiny blobs
            bboxes.append((x, y, x + w, y + h))
    return bboxes

def score_region_with_clip(image_crop: Image.Image, prompt: str):
    inputs = clip_processor(images=image_crop, text=[prompt], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = clip_model(**inputs)
        logits = outputs.logits_per_image  # shape [1,1]
    return logits.item()

def find_best_segment_with_clip(background_image: Image.Image, segmentation_mask: np.ndarray, label_map: dict, prompt: str) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """
    Find the best region (not just label) in the background image that matches the prompt.
    """
    # Extract target label from the prompt
    candidate_labels = [label for label in label_map.values() if label in prompt]
    if not candidate_labels:
        candidate_labels = ["road"]  # fallback

    best_score = float("-inf")
    best_bbox = None

    for label in candidate_labels:
        label_ids = [k for k, v in label_map.items() if v == label]
        if not label_ids:
            continue

        label_id = label_ids[0]
        regions = get_mask_regions(segmentation_mask, label_id)

        for bbox in regions:
            crop = background_image.crop(bbox)
            score = score_region_with_clip(crop, prompt)

            if score > best_score:
                best_score = score
                best_bbox = bbox

    if best_bbox is None:
        raise ValueError(f"No matching region found for prompt: {prompt}")

    return background_image.crop(best_bbox), best_bbox
