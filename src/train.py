import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, classification_report

from text_model import TextSentimentModel


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

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        text = str(row["text_corrected"])
        label = self.label_map[row["sentiment"]]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


def evaluate(model, data_loader, device):
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            predictions = torch.argmax(logits, dim=1)

            all_predictions.extend(
                predictions.cpu().numpy()
            )
            all_labels.extend(
                labels.cpu().numpy()
            )

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
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

    return accuracy, macro_f1, report


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    model_name = "bert-base-uncased"

    # -------------------------
    # Tokenizer
    # -------------------------

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    # -------------------------
    # Dataset
    # -------------------------

    print("Loading datasets...")

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

    # -------------------------
    # DataLoader
    # -------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    print(
        f"Training batches: "
        f"{len(train_loader)}"
    )

    print(
        f"Validation batches: "
        f"{len(val_loader)}"
    )

    # -------------------------
    # Model
    # -------------------------

    print("Loading model...")

    model = TextSentimentModel(
        model_name=model_name,
        num_classes=3,
        dropout=0.3,
    ).to(device)

    print("Model loaded successfully.")

    # -------------------------
    # Optimizer / Loss
    # -------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-5,
    )

    criterion = torch.nn.CrossEntropyLoss()

    epochs = 1

    # Track best model
    best_macro_f1 = 0.0

    # -------------------------
    # Training
    # -------------------------

    for epoch in range(epochs):

        print(
            f"\nStarting Epoch "
            f"{epoch + 1}/{epochs}"
        )

        model.train()

        total_loss = 0.0

        for batch_idx, batch in enumerate(
            train_loader
        ):

            print(
                f"Processing batch "
                f"{batch_idx + 1}/"
                f"{len(train_loader)}"
            )

            input_ids = batch[
                "input_ids"
            ].to(device)

            attention_mask = batch[
                "attention_mask"
            ].to(device)

            labels = batch[
                "label"
            ].to(device)

            optimizer.zero_grad()

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            loss = criterion(
                logits,
                labels,
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % 50 == 0:

                avg_loss = (
                    total_loss
                    / (batch_idx + 1)
                )

                print(
                    f"  Average Loss: "
                    f"{avg_loss:.4f}"
                )

        avg_loss = (
            total_loss
            / len(train_loader)
        )

        print(
            f"\nEpoch {epoch + 1}/{epochs} "
            f"completed"
        )

        print(
            f"Training Loss: "
            f"{avg_loss:.4f}"
        )

        # -------------------------
        # Validation
        # -------------------------

        print("Running validation...")

        (
            val_accuracy,
            val_macro_f1,
            report,
        ) = evaluate(
            model,
            val_loader,
            device,
        )

        print(
            f"Validation Accuracy: "
            f"{val_accuracy:.4f}"
        )

        print(
            f"Validation Macro-F1: "
            f"{val_macro_f1:.4f}"
        )

        print("\nClassification Report:")
        print(report)

        # -------------------------
        # Save best model
        # -------------------------

        if val_macro_f1 > best_macro_f1:

            best_macro_f1 = val_macro_f1

            torch.save(
                model.state_dict(),
                "models/text_model_best.pt",
            )

            print(
                "\nBest model saved to "
                "models/text_model_best.pt"
            )

    print(
        "\nTraining finished successfully."
    )


if __name__ == "__main__":
    main()