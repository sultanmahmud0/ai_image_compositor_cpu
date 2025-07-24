
import numpy as np
from collections import Counter
from transformers import CLIPProcessor, CLIPModel
import torch

# Load CLIP once (optimize for CPU)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_top_segments(mask: np.ndarray, label_map: dict, top_k=3):
    """
    Get top-k most frequent semantic labels from background mask.
    """
    flat_mask = mask.flatten()
    counts = Counter(flat_mask)
    top_labels = counts.most_common(top_k)
    return [label_map[idx] for idx, _ in top_labels]

def rank_segments_with_clip(caption: str, segments: list[str]):
    """
    Use CLIP to find the best segment match for the caption.
    """
    texts = [f"{caption} on {segment}" for segment in segments]
    inputs = clip_processor(text=texts, return_tensors="pt", padding=True)

    with torch.no_grad():
        text_features = clip_model.get_text_features(**inputs)

    # Compare similarity to a neutral query like "this image"
    neutral = clip_processor(text=[caption], return_tensors="pt", padding=True)
    with torch.no_grad():
        caption_features = clip_model.get_text_features(**neutral)

    # Cosine similarity
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    caption_features = caption_features / caption_features.norm(p=2, dim=-1, keepdim=True)
    similarities = (text_features @ caption_features.T).squeeze()

    best_idx = similarities.argmax().item()
    return segments[best_idx]

def generate_prompt(caption: str, background_mask: np.ndarray, label_map: dict):
    """
    Combines object caption with the most suitable background label.
    """
    top_segments = get_top_segments(background_mask, label_map, top_k=5)
    best_segment = rank_segments_with_clip(caption, top_segments)
    return f"{caption} on {best_segment}"
