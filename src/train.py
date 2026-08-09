import os
import random

import numpy as np
import pandas as pd
import torch

from torch.utils.data import (
    Dataset,
    DataLoader,
    WeightedRandomSampler,
)

from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

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

        self.label_map = {
            "negative": 0,
            "neutral": 1,
            "positive": 2,
        }

        # -------------------------------------------------
        # Check missing text
        # -------------------------------------------------

        missing_count = self.df["text_corrected"].isna().sum()

        if missing_count > 0:
            print(
                f"Warning: {missing_count} rows "
                f"have missing text."
            )

            self.df["text_corrected"] = (
                self.df["text_corrected"]
                .fillna("")
            )

        # -------------------------------------------------
        # Check labels
        # -------------------------------------------------

        invalid_labels = (
            set(self.df["sentiment"].unique())
            - set(self.label_map.keys())
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

        label = self.label_map[
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
# Evaluation
# =========================================================

def evaluate(
    model,
    data_loader,
    device,
):

    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for batch in data_loader:

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

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

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

    cm = confusion_matrix(
        all_labels,
        all_predictions,
    )

    # -----------------------------------------------------
    # Prediction distribution
    # -----------------------------------------------------

    prediction_counts = np.bincount(
        all_predictions,
        minlength=3,
    )

    return (
        accuracy,
        macro_f1,
        report,
        cm,
        prediction_counts,
    )


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
    # Configuration
    # =====================================================

    model_name = "bert-base-uncased"

    max_length = 64

    batch_size = 4

    # IMPORTANT:
    # Increased from 1e-5 to 2e-5
    learning_rate = 2e-5

    epochs = 5

    patience = 2

    weight_decay = 0.01

    warmup_ratio = 0.1

    # =====================================================
    # Tokenizer
    # =====================================================

    print(
        "\nLoading tokenizer..."
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    # =====================================================
    # Dataset
    # =====================================================

    print(
        "\nLoading datasets..."
    )

    train_dataset = TextDataset(
        "data/processed/train.csv",
        tokenizer,
        max_length=max_length,
    )

    val_dataset = TextDataset(
        "data/processed/val.csv",
        tokenizer,
        max_length=max_length,
    )

    print(
        f"Training samples: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation samples: "
        f"{len(val_dataset)}"
    )

    # =====================================================
    # Label Distribution
    # =====================================================

    print(
        "\nTraining label distribution:"
    )

    print(
        train_dataset.df[
            "sentiment"
        ].value_counts()
    )

    print(
        "\nTraining label percentage:"
    )

    print(
        train_dataset.df[
            "sentiment"
        ].value_counts(
            normalize=True
        )
    )

    print(
        "\nValidation label distribution:"
    )

    print(
        val_dataset.df[
            "sentiment"
        ].value_counts()
    )

    print(
        "\nValidation label percentage:"
    )

    print(
        val_dataset.df[
            "sentiment"
        ].value_counts(
            normalize=True
        )
    )

    # =====================================================
    # WeightedRandomSampler
    # =====================================================

    print(
        "\nCreating WeightedRandomSampler..."
    )

    train_labels = np.array(
        [
            train_dataset.label_map[label]
            for label
            in train_dataset.df[
                "sentiment"
            ]
        ]
    )

    class_counts = np.bincount(
        train_labels,
        minlength=3,
    )

    print(
        "\nClass counts:"
    )

    print(
        f"negative: {class_counts[0]}"
    )

    print(
        f"neutral:  {class_counts[1]}"
    )

    print(
        f"positive: {class_counts[2]}"
    )

    # -----------------------------------------------------
    # Square-root inverse frequency
    #
    # Less aggressive than 1 / class_counts
    # -----------------------------------------------------

    class_weights = (
        1.0 / np.sqrt(class_counts)
    )

    print(
        "\nSampler class weights:"
    )

    print(
        f"negative: "
        f"{class_weights[0]:.6f}"
    )

    print(
        f"neutral:  "
        f"{class_weights[1]:.6f}"
    )

    print(
        f"positive: "
        f"{class_weights[2]:.6f}"
    )

    sample_weights = torch.tensor(
        [
            class_weights[label]
            for label in train_labels
        ],
        dtype=torch.double,
    )

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(
            sample_weights
        ),
        replacement=True,
    )

    # =====================================================
    # DataLoader
    # =====================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    print(
        f"\nTraining batches: "
        f"{len(train_loader)}"
    )

    print(
        f"Validation batches: "
        f"{len(val_loader)}"
    )

    # =====================================================
    # Model
    # =====================================================

    print(
        "\nLoading model..."
    )

    model = TextSentimentModel(
        model_name=model_name,
        num_classes=3,
        dropout=0.3,
    ).to(device)

    print(
        "Model loaded successfully."
    )

    # =====================================================
    # Loss
    # =====================================================

    # IMPORTANT:
    #
    # NO class weights here.
    #
    # WeightedRandomSampler already handles
    # class imbalance.
    #
    # Using both sampler + class-weighted loss
    # can over-correct minority classes.
    # =====================================================

    criterion = torch.nn.CrossEntropyLoss()

    print(
        "\nLoss function:"
    )

    print(
        "CrossEntropyLoss "
        "(no class weights)"
    )

    # =====================================================
    # Optimizer
    # =====================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # =====================================================
    # Scheduler
    # =====================================================

    total_steps = (
        len(train_loader)
        * epochs
    )

    warmup_steps = int(
        total_steps * warmup_ratio
    )

    scheduler = (
        get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
    )

    print(
        "\nTraining configuration:"
    )

    print(
        f"Learning rate: "
        f"{learning_rate}"
    )

    print(
        f"Batch size: "
        f"{batch_size}"
    )

    print(
        f"Epochs: "
        f"{epochs}"
    )

    print(
        f"Weight decay: "
        f"{weight_decay}"
    )

    print(
        f"Warmup ratio: "
        f"{warmup_ratio}"
    )

    print(
        f"Warmup steps: "
        f"{warmup_steps}"
    )

    print(
        f"Total training steps: "
        f"{total_steps}"
    )

    # =====================================================
    # Best Model / Early Stopping
    # =====================================================

    os.makedirs(
        "models",
        exist_ok=True,
    )

    best_macro_f1 = 0.0

    early_stop_counter = 0

    # =====================================================
    # Training Loop
    # =====================================================

    for epoch in range(epochs):

        print(
            "\n"
            + "=" * 60
        )

        print(
            f"Starting Epoch "
            f"{epoch + 1}/{epochs}"
        )

        print(
            "=" * 60
        )

        model.train()

        total_loss = 0.0

        # -------------------------------------------------
        # Training
        # -------------------------------------------------

        for batch_idx, batch in enumerate(
            train_loader
        ):

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

            optimizer.zero_grad()

            # Forward
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            # Loss
            loss = criterion(
                logits,
                labels,
            )

            # Backward
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            # Update
            optimizer.step()

            scheduler.step()

            total_loss += loss.item()

            # -------------------------------------------------
            # Progress
            # -------------------------------------------------

            if (
                batch_idx + 1
            ) % 50 == 0:

                avg_loss = (
                    total_loss
                    / (batch_idx + 1)
                )

                current_lr = (
                    optimizer.param_groups[0]["lr"]
                )

                print(
                    f"Batch "
                    f"{batch_idx + 1}/"
                    f"{len(train_loader)} "
                    f"| Loss: "
                    f"{avg_loss:.4f} "
                    f"| LR: "
                    f"{current_lr:.2e}"
                )

        # =================================================
        # Epoch Loss
        # =================================================

        avg_loss = (
            total_loss
            / len(train_loader)
        )

        print(
            f"\nEpoch "
            f"{epoch + 1}/{epochs} "
            f"completed"
        )

        print(
            f"Training Loss: "
            f"{avg_loss:.4f}"
        )

        print(
            f"Final Epoch LR: "
            f"{optimizer.param_groups[0]['lr']:.2e}"
        )

        # =================================================
        # Validation
        # =================================================

        print(
            "\nRunning validation..."
        )

        (
            val_accuracy,
            val_macro_f1,
            report,
            cm,
            prediction_counts,
        ) = evaluate(
            model,
            val_loader,
            device,
        )

        # -------------------------------------------------
        # Prediction Distribution
        # -------------------------------------------------

        print(
            "\nPrediction distribution:"
        )

        print(
            f"negative: "
            f"{prediction_counts[0]}"
        )

        print(
            f"neutral:  "
            f"{prediction_counts[1]}"
        )

        print(
            f"positive: "
            f"{prediction_counts[2]}"
        )

        # -------------------------------------------------
        # Metrics
        # -------------------------------------------------

        print(
            f"\nValidation Accuracy: "
            f"{val_accuracy:.4f}"
        )

        print(
            f"Validation Macro-F1: "
            f"{val_macro_f1:.4f}"
        )

        print(
            "\nClassification Report:"
        )

        print(report)

        # =================================================
        # Confusion Matrix
        # =================================================

        print(
            "\nConfusion Matrix:"
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

        # =================================================
        # Save Best Model
        # =================================================

        if (
            val_macro_f1
            > best_macro_f1
        ):

            best_macro_f1 = (
                val_macro_f1
            )

            torch.save(
                model.state_dict(),
                "models/text_model_best.pt",
            )

            print(
                "\nBest model saved to "
                "models/text_model_best.pt"
            )

            early_stop_counter = 0

        else:

            early_stop_counter += 1

            print(
                "\nNo improvement."
            )

            print(
                f"Early stopping counter: "
                f"{early_stop_counter}/"
                f"{patience}"
            )

            if (
                early_stop_counter
                >= patience
            ):

                print(
                    "\nEarly stopping "
                    "triggered."
                )

                break

    # =====================================================
    # Finished
    # =====================================================

    print(
        "\n"
        + "=" * 60
    )

    print(
        "Training finished successfully."
    )

    print(
        f"Best Validation Macro-F1: "
        f"{best_macro_f1:.4f}"
    )

    print(
        "=" * 60
    )


# =========================================================
# Run
# =========================================================

if __name__ == "__main__":
    main()