import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pandas as pd

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

    correct = 0
    total = 0

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

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total if total > 0 else 0

    return accuracy


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

    tokenizer = AutoTokenizer.from_pretrained(model_name)

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

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

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

    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

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

    # CPU-friendly setting
    epochs = 1

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

        for batch_idx, batch in enumerate(train_loader):

            print(
                f"Processing batch "
                f"{batch_idx + 1}/{len(train_loader)}"
            )

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

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
                avg_loss = total_loss / (batch_idx + 1)

                print(
                    f"  Average Loss: "
                    f"{avg_loss:.4f}"
                )

        avg_loss = total_loss / len(train_loader)

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

        val_accuracy = evaluate(
            model,
            val_loader,
            device,
        )

        print(
            f"Validation Accuracy: "
            f"{val_accuracy:.4f}"
        )

    print("\nTraining finished successfully.")


if __name__ == "__main__":
    main()