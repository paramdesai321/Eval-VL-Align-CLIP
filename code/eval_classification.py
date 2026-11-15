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
    

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    local_dir = "./clip"
    model = CLIPModel.from_pretrained(local_dir).to(device)
    preprocessor = CLIPProcessor.from_pretrained(local_dir)
    model.eval()

    num_classes  = len(val_dataset.id2label)
    class_labels = list(val_dataset.id2label.values())
    prompts = [f"this is an image mislabeled as {label}" for label in class_labels]

    text_inputs = preprocessor(text=prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_embeds = model.get_text_features(**text_inputs).pooler_output
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    correct = 0
    total = len(val_dataset)

    with torch.no_grad():
        for image, label, _ in val_dataset:
            image_inputs = preprocessor(images=image, return_tensors="pt").to(device)
            image_embed = model.get_image_features(**image_inputs).pooler_output
            image_embed = image_embed / image_embed.norm(dim=-1, keepdim=True)

            similarity = image_embed @ text_embeds.T
            probs = similarity.softmax(dim=-1)

            pred_idx = probs.argmax(dim=-1).item()
            pred_label = pred_label = class_labels[pred_idx]

            if pred_label == label:
                correct += 1

    accuracy = (correct / total) * 100
    print(f"Overall Validation Accuracy: {accuracy:.2f}% ({correct}/{total})")
if __name__ == "__main__":
    main()
