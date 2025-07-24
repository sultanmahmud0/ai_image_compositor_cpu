
import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet101
from PIL import Image
from pathlib import Path

# Initialize model once
model = deeplabv3_resnet101(pretrained=True)
model.eval().to("cpu")

# Simplified label mapping for demonstration (extend as needed)
COCO_LABELS = {
    0: "background", 1: "aeroplane", 2: "bicycle", 3: "bird", 4: "boat",
    5: "bottle", 6: "bus", 7: "car", 8: "cat", 9: "chair", 10: "cow",
    11: "dining table", 12: "dog", 13: "horse", 14: "motorbike", 15: "person",
    16: "potted plant", 17: "sheep", 18: "sofa", 19: "train", 20: "tv monitor",
    21: "road", 22: "sidewalk", 23: "grass", 24: "river", 25: "sea", 26: "floor"
}

def segment_background(image_path, output_dir="outputs"):
    print("Segmenting background using DeepLabV3+...")

    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = transform(image).unsqueeze(0).to("cpu")

    # Inference
    with torch.no_grad():
        output = model(input_tensor)["out"][0]
    predicted = torch.argmax(output, dim=0).byte().cpu().numpy()

    # Save raw label map as image
    seg_map = Image.fromarray(predicted)
    seg_path = output_dir / f"{image_path.stem}_segmap.png"
    seg_map.save(seg_path)

    print(f"Saved segmentation map to: {seg_path}")
    return predicted, COCO_LABELS
