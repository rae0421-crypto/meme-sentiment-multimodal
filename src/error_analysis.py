import os
import random

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from text_model import TextSentimentModel


# =========================================================
# Reproducibility
# =========================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =========================================================
# Configuration
# =========================================================

MODEL_NAME = "bert-base-uncased"

VAL_PATH = "data/processed/val.csv"

MODEL_PATH = "models/text_model_best.pt"

OUTPUT_PATH = "data/error_analysis.csv"

MAX_LENGTH = 64

BATCH_SIZE = 8


LABEL_MAP = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}

ID_TO_LABEL = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


# =========================================================
# Dataset
# =========================================================

class TextDataset(Dataset):

    def __init__(
        self,
        csv_path,
        tokenizer,
        max_length=64,
    ):

        self.df = pd.read_csv(csv_path)

        self.tokenizer = tokenizer

        self.max_length = max_length

        # Missing text
        missing_count = self.df["text_corrected"].isna().sum()

        if missing_count > 0:

            print(
                f"Warning: {missing_count} rows have "
                f"missing text."
            )

            self.df["text_corrected"] = (
                self.df["text_corrected"]
                .fillna("")
            )

        # Check labels
        invalid_labels = (
            set(self.df["sentiment"].unique())
            - set(LABEL_MAP.keys())
        )

        if invalid_labels:

            raise ValueError(
                f"Unknown sentiment labels: "
                f"{invalid_labels}"
            )

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        text = str(
            row["text_corrected"]
        )

        label = LABEL_MAP[
            row["sentiment"]
        ]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": (
                encoding["input_ids"]
                .squeeze(0)
            ),

            "attention_mask": (
                encoding["attention_mask"]
                .squeeze(0)
            ),

            "label": torch.tensor(
                label,
                dtype=torch.long,
            ),
        }


# =========================================================
# Main
# =========================================================

def main():

    # =====================================================
    # Device
    # =====================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Using device:",
        device,
    )


    # =====================================================
    # Load tokenizer
    # =====================================================

    print(
        "\nLoading tokenizer..."
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )


    # =====================================================
    # Load validation dataset
    # =====================================================

    print(
        "\nLoading validation dataset..."
    )

    dataset = TextDataset(
        VAL_PATH,
        tokenizer,
        max_length=MAX_LENGTH,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print(
        f"Validation samples: "
        f"{len(dataset)}"
    )


    # =====================================================
    # Load model
    # =====================================================

    print(
        "\nLoading best model..."
    )

    model = TextSentimentModel(
        model_name=MODEL_NAME,
        num_classes=3,
        dropout=0.3,
    ).to(device)

    state_dict = torch.load(
        MODEL_PATH,
        map_location=device,
    )

    model.load_state_dict(
        state_dict
    )

    model.eval()

    print(
        "Best model loaded successfully."
    )


    # =====================================================
    # Prediction
    # =====================================================

    all_predictions = []

    all_labels = []

    all_confidences = []

    print(
        "\nRunning predictions..."
    )

    with torch.no_grad():

        for batch in loader:

            input_ids = (
                batch["input_ids"]
                .to(device)
            )

            attention_mask = (
                batch["attention_mask"]
                .to(device)
            )

            labels = (
                batch["label"]
                .to(device)
            )

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            predictions = torch.argmax(
                probabilities,
                dim=1,
            )

            confidences = torch.max(
                probabilities,
                dim=1,
            ).values

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_confidences.extend(
                confidences.cpu().numpy()
            )


    # =====================================================
    # Metrics
    # =====================================================

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    print(
        "\n"
        + "=" * 60
    )

    print(
        "Validation Results"
    )

    print(
        "=" * 60
    )

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print(
        f"Macro-F1: {macro_f1:.4f}"
    )


    # =====================================================
    # Classification Report
    # =====================================================

    print(
        "\nClassification Report:"
    )

    report = classification_report(
        all_labels,
        all_predictions,
        target_names=[
            "negative",
            "neutral",
            "positive",
        ],
        digits=4,
        zero_division=0,
    )

    print(report)


    # =====================================================
    # Confusion Matrix
    # =====================================================

    cm = confusion_matrix(
        all_labels,
        all_predictions,
    )

    print(
        "Confusion Matrix:"
    )

    print(
        "Rows = Actual"
    )

    print(
        "Columns = Predicted"
    )

    print(
        "          neg  neu  pos"
    )

    print(
        f"negative  "
        f"{cm[0][0]:4d} "
        f"{cm[0][1]:4d} "
        f"{cm[0][2]:4d}"
    )

    print(
        f"neutral   "
        f"{cm[1][0]:4d} "
        f"{cm[1][1]:4d} "
        f"{cm[1][2]:4d}"
    )

    print(
        f"positive  "
        f"{cm[2][0]:4d} "
        f"{cm[2][1]:4d} "
        f"{cm[2][2]:4d}"
    )


    # =====================================================
    # Build Error Analysis DataFrame
    # =====================================================

    analysis_df = dataset.df.copy()

    analysis_df["true_label"] = [
        ID_TO_LABEL[label]
        for label in all_labels
    ]

    analysis_df["predicted_label"] = [
        ID_TO_LABEL[pred]
        for pred in all_predictions
    ]

    analysis_df["confidence"] = (
        np.array(all_confidences)
    )

    analysis_df["correct"] = (
        np.array(all_labels)
        == np.array(all_predictions)
    )


    # =====================================================
    # Error Type
    # =====================================================

    def get_error_type(row):

        if row["correct"]:

            return "correct"

        return (
            row["true_label"]
            + " -> "
            + row["predicted_label"]
        )


    analysis_df["error_type"] = (
        analysis_df.apply(
            get_error_type,
            axis=1,
        )
    )


    # =====================================================
    # Error Summary
    # =====================================================

    errors = analysis_df[
        ~analysis_df["correct"]
    ]

    print(
        "\n"
        + "=" * 60
    )

    print(
        "Error Analysis Summary"
    )

    print(
        "=" * 60
    )

    print(
        f"Total validation samples: "
        f"{len(analysis_df)}"
    )

    print(
        f"Correct predictions: "
        f"{analysis_df['correct'].sum()}"
    )

    print(
        f"Incorrect predictions: "
        f"{len(errors)}"
    )


    print(
        "\nError Types:"
    )

    error_counts = (
        errors["error_type"]
        .value_counts()
    )

    for error_type, count in error_counts.items():

        print(
            f"{error_type:25s}: {count}"
        )


    # =====================================================
    # Important Error Groups
    # =====================================================

    important_errors = [
        "negative -> positive",
        "negative -> neutral",
        "neutral -> positive",
        "neutral -> negative",
        "positive -> neutral",
        "positive -> negative",
    ]

    print(
        "\n"
        + "=" * 60
    )

    print(
        "Important Error Groups"
    )

    print(
        "=" * 60
    )

    for error_type in important_errors:

        subset = analysis_df[
            analysis_df["error_type"]
            == error_type
        ]

        print(
            f"{error_type:25s}: "
            f"{len(subset)}"
        )


    # =====================================================
    # Save Full Error Analysis
    # =====================================================

    os.makedirs(
        "data",
        exist_ok=True,
    )

    analysis_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nFull error analysis saved to:"
    )

    print(
        OUTPUT_PATH
    )


    # =====================================================
    # Print Examples
    # =====================================================

    print(
        "\n"
        + "=" * 60
    )

    print(
        "Representative Error Examples"
    )

    print(
        "=" * 60
    )


    for error_type in important_errors:

        subset = analysis_df[
            analysis_df["error_type"]
            == error_type
        ].copy()

        if len(subset) == 0:

            continue

        print(
            "\n"
            + "-" * 60
        )

        print(
            f"{error_type}"
        )

        print(
            "-" * 60
        )

        # Random sample, maximum 10
        sample_size = min(
            10,
            len(subset),
        )

        examples = subset.sample(
            n=sample_size,
            random_state=SEED,
        )

        for i, (_, row) in enumerate(
            examples.iterrows(),
            start=1,
        ):

            text = str(
                row["text_corrected"]
            )

            # Prevent enormous output
            if len(text) > 300:

                text = (
                    text[:300]
                    + "..."
                )

            print(
                f"\nExample {i}"
            )

            print(
                f"Text: {text}"
            )

            print(
                f"True: "
                f"{row['true_label']}"
            )

            print(
                f"Predicted: "
                f"{row['predicted_label']}"
            )

            print(
                f"Confidence: "
                f"{row['confidence']:.4f}"
            )


    # =====================================================
    # High Confidence Wrong Predictions
    # =====================================================

    print(
        "\n"
        + "=" * 60
    )

    print(
        "High-Confidence Wrong Predictions"
    )

    print(
        "=" * 60
    )

    high_conf_errors = (
        errors
        .sort_values(
            "confidence",
            ascending=False,
        )
        .head(20)
    )

    for i, (_, row) in enumerate(
        high_conf_errors.iterrows(),
        start=1,
    ):

        text = str(
            row["text_corrected"]
        )

        if len(text) > 250:

            text = (
                text[:250]
                + "..."
            )

        print(
            f"\n{i}. "
            f"{row['error_type']} "
            f"(confidence="
            f"{row['confidence']:.4f})"
        )

        print(
            text
        )


    # =====================================================
    # Final
    # =====================================================

    print(
        "\n"
        + "=" * 60
    )

    print(
        "Error analysis completed."
    )

    print(
        "=" * 60
    )


# =========================================================
# Run
# =========================================================

if __name__ == "__main__":

    main()