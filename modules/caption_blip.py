# modules/caption_blip.py

import re
import inflect
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# English article fixer
p = inflect.engine()

def fix_indefinite_article(phrase: str) -> str:
    """
    Correct 'a' vs 'an' at the beginning of a phrase.
    E.g., 'a apple' → 'an apple'
    """
    if phrase.lower().startswith("a "):
        rest = phrase[2:].strip()
        correct_phrase = p.a(rest)
        return correct_phrase[0].upper() + correct_phrase[1:]
    return phrase

def clean_caption(caption: str) -> str:
    """
    Strip background context like 'on a road' or 'in the field'
    and fix grammatical issues.
    """
    caption = caption.strip().lower()

    # Remove trailing scene context (e.g. "on a road", "in the field")
    caption = re.split(r"\b(on|in|at|by|under|near|beside|behind)\b", caption)[0].strip()

    # Capitalize first word
    if caption:
        caption = caption[0].upper() + caption[1:]

    # Fix article ("a apple" → "an apple")
    caption = fix_indefinite_article(caption)

    return caption

def generate_caption(image: Image.Image) -> str:
    """
    Generate and clean caption for the given image.
    """
    image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(**inputs)
    raw_caption = processor.tokenizer.decode(out[0], skip_special_tokens=True)

    return clean_caption(raw_caption)
