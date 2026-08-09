import os
import random
import numpy as np
import torch
import pandas as pd

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
# Random Seed
# =========================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
        # Check columns
        # -------------------------------------------------

        required_columns = [
            "text_corrected",
            "sentiment",
        ]

        for column in required_columns:

            if column not in self.df.columns:

                raise ValueError(
                    f"Missing required column: {column}"
                )

        # -------------------------------------------------
        # Remove missing labels
        # -------------------------------------------------

        self.df = self.df.dropna(
            subset=["sentiment"]
        ).reset_index(drop=True)

        # -------------------------------------------------
        # Handle missing text
        # -------------------------------------------------

        missing_text = (
            self.df["text_corrected"]
            .isna()
            .sum()
        )

        if missing_text > 0:

            print(
                f"Warning: {missing_text} "
                f"rows have missing text."
            )

        self.df["text_corrected"] = (
            self.df["text_corrected"]
            .fillna("")
            .astype(str)
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
            "input_ids": encoding[
                "input_ids"
            ].squeeze(0),

            "attention_mask": encoding[
                "attention_mask"
            ].squeeze(0),

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

            input_ids = batch[
                "input_ids"
            ].to(device)

            attention_mask = batch[
                "attention_mask"
            ].to(device)

            labels = batch[
                "label"
            ].to(device)

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
        labels=[0, 1, 2],
    )

    # -----------------------------------------------------
    # Prediction distribution
    # -----------------------------------------------------

    prediction_counts = np.bincount(
        all_predictions,
        minlength=3,
    )

    print(
        "\nPrediction distribution:"
    )

    print(
        f"negative: {prediction_counts[0]}"
    )

    print(
        f"neutral:  {prediction_counts[1]}"
    )

    print(
        f"positive: {prediction_counts[2]}"
    )

    return (
        accuracy,
        macro_f1,
        report,
        cm,
    )


# =========================================================
# Main
# =========================================================

def main():

    # -----------------------------------------------------
    # Seed
    # -----------------------------------------------------

    set_seed(42)

    # -----------------------------------------------------
    # Device
    # -----------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Using device:",
        device,
    )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model_name = "bert-base-uncased"

    # -----------------------------------------------------
    # Tokenizer
    # -----------------------------------------------------

    print(
        "\nLoading tokenizer..."
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------

    print(
        "\nLoading datasets..."
    )

    train_dataset = TextDataset(
        "data/processed/train.csv",
        tokenizer,
        max_length=64,
    )

    val_dataset = TextDataset(
        "data/processed/val.csv",
        tokenizer,
        max_length=64,
    )

    print(
        f"Training samples: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation samples: "
        f"{len(val_dataset)}"
    )

    # -----------------------------------------------------
    # Label distribution
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # Training labels
    # -----------------------------------------------------

    train_labels = [
        train_dataset.label_map[x]
        for x in train_dataset.df[
            "sentiment"
        ]
    ]

    train_labels_tensor = torch.tensor(
        train_labels,
        dtype=torch.long,
    )

    # -----------------------------------------------------
    # Weighted Random Sampler
    # -----------------------------------------------------

    print(
        "\nCreating WeightedRandomSampler..."
    )

    class_counts = torch.bincount(
        train_labels_tensor,
        minlength=3,
    )

    print(
        "\nClass counts:"
    )

    print(
        f"negative: "
        f"{class_counts[0].item()}"
    )

    print(
        f"neutral:  "
        f"{class_counts[1].item()}"
    )

    print(
        f"positive: "
        f"{class_counts[2].item()}"
    )

    # Inverse frequency
    sampler_class_weights = (
        1.0
        / class_counts.float()
    )

    print(
        "\nSampler class weights:"
    )

    print(
        f"negative: "
        f"{sampler_class_weights[0].item():.6f}"
    )

    print(
        f"neutral:  "
        f"{sampler_class_weights[1].item():.6f}"
    )

    print(
        f"positive: "
        f"{sampler_class_weights[2].item():.6f}"
    )

    sample_weights = torch.tensor(
        [
            sampler_class_weights[
                label
            ].item()
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

    # -----------------------------------------------------
    # DataLoader
    # -----------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        sampler=sampler,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
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

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # Loss
    # -----------------------------------------------------
    #
    # Mild class weighting.
    #
    # Sampler already handles most of the imbalance.
    # Therefore we use only mild additional weighting.
    #

    class_weights = torch.tensor(
        [1.5, 1.0, 0.8],
        dtype=torch.float,
    ).to(device)

    print(
        "\nLoss class weights:"
    )

    print(
        class_weights
    )

    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights
    )

    # -----------------------------------------------------
    # Optimizer
    # -----------------------------------------------------

    learning_rate = 2e-5

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
    )

    # -----------------------------------------------------
    # Training settings
    # -----------------------------------------------------

    epochs = 5

    total_steps = (
        len(train_loader)
        * epochs
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(
            total_steps * 0.1
        ),
        num_training_steps=total_steps,
    )

    # -----------------------------------------------------
    # Best model
    # -----------------------------------------------------

    best_macro_f1 = -1.0

    patience = 2

    epochs_without_improvement = 0

    os.makedirs(
        "models",
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------

    for epoch in range(epochs):

        print(
            f"\n{'=' * 60}"
        )

        print(
            f"Starting Epoch "
            f"{epoch + 1}/{epochs}"
        )

        print(
            f"{'=' * 60}"
        )

        model.train()

        total_loss = 0.0

        # -------------------------------------------------
        # Training loop
        # -------------------------------------------------

        for (
            batch_idx,
            batch,
        ) in enumerate(
            train_loader
        ):

            input_ids = batch[
                "input_ids"
            ].to(device)

            attention_mask = batch[
                "attention_mask"
            ].to(device)

            labels = batch[
                "label"
            ].to(device)

            # ---------------------------------------------
            # Forward
            # ---------------------------------------------

            optimizer.zero_grad()

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            loss = criterion(
                logits,
                labels,
            )

            # ---------------------------------------------
            # Backward
            # ---------------------------------------------

            loss.backward()

            # ---------------------------------------------
            # Gradient clipping
            # ---------------------------------------------

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            # ---------------------------------------------
            # Optimizer
            # ---------------------------------------------

            optimizer.step()

            scheduler.step()

            total_loss += loss.item()

            # ---------------------------------------------
            # Progress
            # ---------------------------------------------

            if (
                batch_idx + 1
            ) % 50 == 0:

                avg_loss = (
                    total_loss
                    / (
                        batch_idx + 1
                    )
                )

                print(
                    f"Batch "
                    f"{batch_idx + 1}/"
                    f"{len(train_loader)} "
                    f"| Loss: "
                    f"{avg_loss:.4f}"
                )

        # -------------------------------------------------
        # Epoch loss
        # -------------------------------------------------

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

        # -------------------------------------------------
        # Validation
        # -------------------------------------------------

        print(
            "\nRunning validation..."
        )

        (
            val_accuracy,
            val_macro_f1,
            report,
            cm,
        ) = evaluate(
            model,
            val_loader,
            device,
        )

        print(
            f"\nValidation Accuracy: "
            f"{val_accuracy:.4f}"
        )

        print(
            f"Validation Macro-F1: "
            f"{val_macro_f1:.4f}"
        )

        # -------------------------------------------------
        # Classification report
        # -------------------------------------------------

        print(
            "\nClassification Report:"
        )

        print(report)

        # -------------------------------------------------
        # Confusion matrix
        # -------------------------------------------------

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

        # -------------------------------------------------
        # Save best model
        # -------------------------------------------------

        if (
            val_macro_f1
            > best_macro_f1
        ):

            best_macro_f1 = (
                val_macro_f1
            )

            epochs_without_improvement = 0

            torch.save(
                model.state_dict(),
                "models/text_model_best.pt",
            )

            print(
                "\nBest model saved to "
                "models/text_model_best.pt"
            )

        else:

            epochs_without_improvement += 1

            print(
                "\nNo improvement."
            )

            print(
                f"Early stopping counter: "
                f"{epochs_without_improvement}/"
                f"{patience}"
            )

            if (
                epochs_without_improvement
                >= patience
            ):

                print(
                    "\nEarly stopping triggered."
                )

                break

    # -----------------------------------------------------
    # Finished
    # -----------------------------------------------------

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