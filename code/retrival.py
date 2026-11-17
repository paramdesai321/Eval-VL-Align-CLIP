import os
import glob
import torch
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel


def precompute_coco_embeddings(
    image_dir: str,
    model: CLIPModel,
    processor: CLIPProcessor,
    device: str,
    cache_path: str = "coco_embeddings.pt",
    batch_size: int = 32,
):
    if os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}...")
        data = torch.load(cache_path, map_location=device)
        return data["embeddings"].to(device), data["image_paths"]

    extensions = ("*.jpg", "*.jpeg", "*.png")
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
    image_paths = sorted(image_paths)

    if len(image_paths) == 0:
        raise FileNotFoundError(f"No images found in {image_dir}. Check your DATA path.")

    print(f"Pre-computing embeddings for {len(image_paths)} COCO images...")
    all_embeddings = []
    valid_paths = []

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size)):
            batch_paths = image_paths[i : i + batch_size]
            batch_images = []

            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    batch_images.append(img)
                    valid_paths.append(p)
                except Exception as e:
                    print(f"Skipping corrupted image {p}: {e}")

            if not batch_images:
                continue

            inputs = processor(images=batch_images, return_tensors="pt", padding=True).to(device)
            image_embeds = model.get_image_features(**inputs).pooler_output
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            all_embeddings.append(image_embeds.cpu())

    image_embeddings = torch.cat(all_embeddings, dim=0)  # [N, embed_dim]

    torch.save({"embeddings": image_embeddings, "image_paths": valid_paths}, cache_path)
    print(f"Saved {image_embeddings.shape[0]} embeddings to {cache_path}.")

    return image_embeddings.to(device), valid_paths

