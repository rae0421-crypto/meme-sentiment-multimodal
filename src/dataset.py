from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class MemotionDataset(Dataset):
    def __init__(self, csv_path, image_dir, tokenizer=None, transform=None):
        self.df = pd.read_csv(csv_path)

        # Merge labels into 3 classes
        sentiment_map = {
            "very_positive": "positive",
            "positive": "positive",
            "neutral": "neutral",
            "negative": "negative",
            "very_negative": "negative",
        }

        self.df["label"] = self.df["overall_sentiment"].map(sentiment_map)

        label_to_id = {
            "negative": 0,
            "neutral": 1,
            "positive": 2,
        }

        self.df["label_id"] = self.df["label"].map(label_to_id)

        self.image_dir = Path(image_dir)
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = Image.open(
            self.image_dir / row["image_name"]
        ).convert("RGB")

        if self.transform:
            image = self.transform(image)

        text = row["text_corrected"]

        label = int(row["label_id"])

        return {
            "image": image,
            "text": text,
            "label": label,
        }