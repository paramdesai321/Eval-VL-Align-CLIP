import torch
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel


from dataset import ImagenetteDataset

def main():
    txt_file_path = './DATA/imagenette2/val.txt'
    image_root_dir = './DATA/imagenette2'
    val_dataset = ImagenetteDataset(txt_file_path, image_root_dir)
    

    sample_img, sample_label, img_path = val_dataset[0]
    print(f"Sample 0 -> Image type: {type(sample_img)}, Label: {sample_label}")
    

    device = 'mps' if torch.mps_is_available() else 'cpu'
    local_dir = "./clip"
    with torch.no_grad():
        for image, label, _ in val_dataset:
            inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True).to(model.device)
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image  # [1, 10] cosine similarity scores (scaled)
            probs = logits_per_image.softmax(dim=1)

            pred_idx = probs.argmax(dim=1).item()
            if class_names[pred_idx] == label:
                correct += 1

    accuracy = (correct / total) * 100
    print(f"Zero-Shot Classification Accuracy: {accuracy:.2f}% ({correct}/{total})")

if __name__ == "__main__":
    main()
