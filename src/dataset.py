from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class MemotionDataset(Dataset):
    def __init__(
        self,
        csv_path,
        image_dir,
        tokenizer=None,
        transform=None,
    ):
        self.df = pd.read_csv(csv_path)

        # =====================================================
        # Merge sentiment labels into 3 classes
        # =====================================================

        sentiment_map = {
            "very_positive": "positive",
            "positive": "positive",
            "neutral": "neutral",
            "negative": "negative",
            "very_negative": "negative",
        }

        self.df["label"] = self.df[
            "overall_sentiment"
        ].map(sentiment_map)

        label_to_id = {
            "negative": 0,
            "neutral": 1,
            "positive": 2,
        }

        self.df["label_id"] = self.df[
            "label"
        ].map(label_to_id)

        # =====================================================
        # Paths / tokenizer / transform
        # =====================================================

        self.image_dir = Path(image_dir)

        self.tokenizer = tokenizer

        self.transform = transform

    # =========================================================
    # Dataset length
    # =========================================================

    def __len__(self):
        return len(self.df)

    # =========================================================
    # Get item
    # =========================================================

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        # -----------------------------------------------------
        # Image
        # -----------------------------------------------------

        image_path = (
            self.image_dir
            / str(row["image_name"])
        )

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

        except Exception as e:

            print(
                f"\nWarning: corrupted image skipped:"
            )

            print(
                f"Image: {image_path}"
            )

            print(
                f"Error: {e}"
            )

            # Try the next sample
            new_idx = (
                (idx + 1)
                % len(self.df)
            )

            return self.__getitem__(
                new_idx
            )

        # -----------------------------------------------------
        # Image transform
        # -----------------------------------------------------

        if self.transform:

            image = self.transform(
                image
            )

        # -----------------------------------------------------
        # Text
        # -----------------------------------------------------

        text = row[
            "text_corrected"
        ]

        # Handle NaN / missing text
        if pd.isna(text):

            text = ""

        else:

            text = str(text)

        # -----------------------------------------------------
        # Label
        # -----------------------------------------------------

        label = int(
            row["label_id"]
        )

        # -----------------------------------------------------
        # Return
        # -----------------------------------------------------

        return {
            "image": image,
            "text": text,
            "label": label,
        }