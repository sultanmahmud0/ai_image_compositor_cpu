# Entry point

from modules.sam_segment import segment_object
from modules.caption_blip import generate_caption
from modules.semantic_segmentation import segment_background
from modules.depth_estimation import estimate_depth
from modules.generate_prompt import generate_prompt
from modules.clip_match import find_best_segment_with_clip
from modules.object_placer import (
    get_average_depth,
    calculate_scale,
    resize_object,
    blend_object
)
from modules.util import get_rgb_image_from_path

import warnings
warnings.filterwarnings("ignore", message="Some weights of .* were not initialized")

def run_pipeline(object_path, background_path):

    background_image = get_rgb_image_from_path(background_path)
    object_image = get_rgb_image_from_path(object_path)

    print("🔹 Step 1: Segmenting object...")
    object_cropped, object_binary_mask = segment_object(object_path)

    print("🔹 Step 2: Generating caption...")
    object_caption = generate_caption(object_image)
    print("Caption:", object_caption)

    print("🔹 Step 3: Segmenting background (DeepLabV3+)...")
    background_mask, label_map = segment_background(background_img_path)

    print("🔹 Step 4: Estimating depth map (MiDaS)...")
    depth_map  = estimate_depth(background_path)

    print("🔹 Step 5: Final prompt generation using CLIP...")
    final_prompt = generate_prompt(object_caption, background_mask, label_map)
    print("Final Prompt:", final_prompt)

    print("🔹 Step 6: Find the best suitable place in the background image using the prompt...")
    _, placement_bbox = find_best_segment_with_clip(background_image, background_mask, label_map, final_prompt)

    print("🔹 Step 7: Resize the object to fit the best suitable region...")
    avg_depth = get_average_depth(depth_map, placement_bbox)
    original_size = object_cropped.size
    new_size = calculate_scale(original_size, avg_depth)
    resized_object = resize_object(object_cropped, new_size)
    resized_mask = object_binary_mask.resize(new_size)

    print("🔹 Step 8: Place and blend the object into the background image...")
    x1, y1, x2, y2 = placement_bbox
    placement_pos = (x1 + ((x2 - x1 - new_size[0]) // 2), y1 + ((y2 - y1 - new_size[1]) // 2))
    final_image = blend_object(background_image, resized_object, resized_mask, placement_pos)

    final_image.save("outputs/composed.png")
    print("✅ Final composed image saved.")

    print("✅ Pipeline steps complete.")

if __name__ == "__main__":
    object_img_path = "inputs/object.png"
    background_img_path = "inputs/background.jpg"
    run_pipeline(object_img_path, background_img_path)



