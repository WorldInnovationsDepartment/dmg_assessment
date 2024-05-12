import matplotlib.pyplot as plt
import random
from transformers import ViTForImageClassification, AutoFeatureExtractor
import torch
from PIL import Image
import os
from train import WindowDataset


def visualize_predictions(dataset, model, feature_extractor, num_images=5):
    broken_indices = [i for i, item in enumerate(dataset.data) if item['category_id'] == 1]
    not_broken_indices = [i for i, item in enumerate(dataset.data) if item['category_id'] == 2]

    selected_broken = random.sample(broken_indices, num_images)
    selected_not_broken = random.sample(not_broken_indices, num_images)

    fig, axs = plt.subplots(2, num_images, figsize=(15, 6))
    for ax, idx in zip(axs[0], selected_broken):
        image, true_label = load_image_and_label(dataset, idx)
        pred_label = predict_image(model, feature_extractor, image)
        ax.imshow(image)
        ax.set_title(f"True: Broken\nPred: {'Broken' if pred_label == 0 else 'Not broken'}")
        ax.axis('off')
    
    for ax, idx in zip(axs[1], selected_not_broken):
        image, true_label = load_image_and_label(dataset, idx)
        pred_label = predict_image(model, feature_extractor, image)
        ax.imshow(image)
        ax.set_title(f"True: Not Broken\nPred: {'Broken' if pred_label == 0 else 'Not broken'}")
        ax.axis('off')
    
    plt.show()

def load_image_and_label(dataset, idx):
    item = dataset.data[idx]
    img_path = os.path.join(dataset.img_dir, item['file_name'])
    image = Image.open(img_path).convert("RGB")
    return image, item['category_id'] - 1

def predict_image(model, feature_extractor, image):
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        preds = outputs.logits.argmax(-1)
    return preds.item()


if __name__ == "__main__":
    model = ViTForImageClassification.from_pretrained("./results/checkpoint-4700")
    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/dino-vits8")
    test_dataset = WindowDataset(json_file='dataset/test_config.json', img_dir='dataset/test', feature_extractor=feature_extractor)

    visualize_predictions(test_dataset, model, feature_extractor, num_images=5)
