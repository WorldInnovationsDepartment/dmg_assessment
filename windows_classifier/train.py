from transformers import ViTForImageClassification, AutoFeatureExtractor, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch
from PIL import Image
import os
import json
from torch.utils.data.dataloader import default_collate
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import numpy as np

def data_collator(batch):
    batch = {k: [dic[k] for dic in batch] for k in batch[0]}
    batch = {k: default_collate(v) for k, v in batch.items()}
    return batch

class WindowDataset(Dataset):
    def __init__(self, json_file, img_dir, feature_extractor):
        self.img_dir = img_dir
        self.feature_extractor = feature_extractor
        
        with open(json_file) as f:
            self.data = json.load(f)
        
        self.labels = {1: "Broken", 2: "Not broken"}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.img_dir, item['file_name'])
        image = Image.open(img_path).convert("RGB")
        label = item['category_id'] - 1  

        encoded_inputs = self.feature_extractor(images=image, return_tensors='pt')
        
        for k, v in encoded_inputs.items():
            encoded_inputs[k] = v.squeeze(0)
        
        encoded_inputs['labels'] = torch.tensor(label)
        
        return encoded_inputs


if __name__ == "__main__":
    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/dino-vits8")

    train_dataset = WindowDataset(
        json_file='dataset/train_config.json', 
        img_dir='dataset/train', 
        feature_extractor=feature_extractor
        )

    model = ViTForImageClassification.from_pretrained("facebook/dino-vits8", num_labels=2)

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=8,
        num_train_epochs=100,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        do_train=True,
        do_eval=False,
        remove_unused_columns=False,
        save_strategy='epoch', 
        save_total_limit=3
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=feature_extractor
    )

    trainer.train()

    test_dataset = WindowDataset(
        json_file='dataset/test_config.json', 
        img_dir='dataset/test', 
        feature_extractor=feature_extractor
    )

    predictions = trainer.predict(test_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=-1)
    true_labels = [item['labels'].item() for item in test_dataset]

    print("Precision: {:.2f}".format(precision_score(true_labels, pred_labels, average='macro')))
    print("Recall: {:.2f}".format(recall_score(true_labels, pred_labels, average='macro')))
    print("F1-Score: {:.2f}".format(f1_score(true_labels, pred_labels, average='macro')))
    print("\nFull classification report:\n", classification_report(true_labels, pred_labels))