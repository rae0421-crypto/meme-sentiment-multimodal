from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torchvision import transforms

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

from dataset import MemotionDataset
from multimodal_model import MultimodalSentimentModel


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_CSV = (
    BASE_DIR
    / "data"
    / "processed"
    / "train.csv"
)

VAL_CSV = (
    BASE_DIR
    / "data"
    / "processed"
    / "val.csv"
)

IMAGE_DIR = (
    BASE_DIR
    / "data"
    / "raw"
    / "memotion_dataset_7k"
    / "images"
)

MODEL_DIR = (
    BASE_DIR
    / "models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_SAVE_PATH = (
    MODEL_DIR
    / "best_multimodal_v2.pt"
)


# ============================================================
# Hyperparameters
# ============================================================

TEXT_MODEL_NAME = (
    "distilbert-base-uncased"
)

NUM_CLASSES = 3

MAX_LENGTH = 64

BATCH_SIZE = 16

EPOCHS = 8

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0

PATIENCE = 2


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    f"Using device: {device}"
)


# ============================================================
# Check files
# ============================================================

print("\nChecking files...")

if not TRAIN_CSV.exists():
    raise FileNotFoundError(
        f"Training CSV not found: {TRAIN_CSV}"
    )

if not VAL_CSV.exists():
    raise FileNotFoundError(
        f"Validation CSV not found: {VAL_CSV}"
    )

if not IMAGE_DIR.exists():
    raise FileNotFoundError(
        f"Image directory not found: {IMAGE_DIR}"
    )

print(
    f"Train CSV: {TRAIN_CSV}"
)

print(
    f"Val CSV:   {VAL_CSV}"
)

print(
    f"Images:    {IMAGE_DIR}"
)


# ============================================================
# Tokenizer
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    TEXT_MODEL_NAME
)


# ============================================================
# Image transforms
# ============================================================

train_transform = transforms.Compose(
    [
        transforms.Resize(
            (224, 224)
        ),

        transforms.RandomHorizontalFlip(
            p=0.5
        ),

        transforms.RandomRotation(
            5
        ),

        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.15,
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ]
)


val_transform = transforms.Compose(
    [
        transforms.Resize(
            (224, 224)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ]
)


# ============================================================
# Dataset
# ============================================================

print("\nLoading datasets...")

train_dataset = MemotionDataset(
    csv_path=TRAIN_CSV,
    image_dir=IMAGE_DIR,
    tokenizer=tokenizer,
    transform=train_transform,
)

val_dataset = MemotionDataset(
    csv_path=VAL_CSV,
    image_dir=IMAGE_DIR,
    tokenizer=tokenizer,
    transform=val_transform,
)

print(
    f"Training samples: "
    f"{len(train_dataset)}"
)

print(
    f"Validation samples: "
    f"{len(val_dataset)}"
)


# ============================================================
# Class distribution
# ============================================================

print(
    "\nChecking class distribution..."
)

train_labels = (
    train_dataset.df["label_id"]
)

label_counts = (
    train_labels
    .value_counts()
    .sort_index()
)

label_names = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


for label_id in range(
    NUM_CLASSES
):

    count = label_counts.get(
        label_id,
        0,
    )

    print(
        f"{label_id} "
        f"({label_names[label_id]}): "
        f"{count}"
    )


# ============================================================
# Class weights
# ============================================================

total_samples = len(
    train_labels
)

class_weights = []

for label_id in range(
    NUM_CLASSES
):

    count = label_counts.get(
        label_id,
        0,
    )

    if count == 0:

        weight = 0.0

    else:

        weight = (
            total_samples
            / (
                NUM_CLASSES
                * count
            )
        )

    class_weights.append(
        weight
    )


class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32,
).to(device)


print(
    "\nClass weights:"
)

for label_id, weight in enumerate(
    class_weights
):

    print(
        f"{label_names[label_id]}: "
        f"{weight.item():.4f}"
    )


# ============================================================
# DataLoader
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)


# ============================================================
# Model
# ============================================================

print(
    "\nLoading multimodal V2 model..."
)

model = MultimodalSentimentModel(
    text_model_name=TEXT_MODEL_NAME,
    num_classes=NUM_CLASSES,
    freeze_encoders=True,
)

model = model.to(device)


# ============================================================
# Count trainable parameters
# ============================================================

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

total_params = sum(
    p.numel()
    for p in model.parameters()
)

print(
    f"\nTrainable parameters: "
    f"{trainable_params:,}"
)

print(
    f"Total parameters: "
    f"{total_params:,}"
)


# ============================================================
# Loss
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.AdamW(
    filter(
        lambda p: p.requires_grad,
        model.parameters(),
    ),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


# ============================================================
# Training
# ============================================================

def train_one_epoch():

    model.train()

    total_loss = 0.0

    all_labels = []

    all_predictions = []

    for batch_idx, batch in enumerate(
        train_loader
    ):

        images = batch[
            "image"
        ].to(device)

        labels = batch[
            "label"
        ].to(device)

        encoding = tokenizer(
            list(batch["text"]),
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        input_ids = encoding[
            "input_ids"
        ].to(device)

        attention_mask = encoding[
            "attention_mask"
        ].to(device)

        optimizer.zero_grad()

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images,
        )

        loss = criterion(
            logits,
            labels,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        total_loss += loss.item()

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        all_labels.extend(
            labels.detach()
            .cpu()
            .numpy()
        )

        all_predictions.extend(
            predictions.detach()
            .cpu()
            .numpy()
        )

        if (
            (batch_idx + 1)
            % 50
            == 0
        ):

            print(
                f"Batch "
                f"{batch_idx + 1}/"
                f"{len(train_loader)} "
                f"- Loss: "
                f"{loss.item():.4f}"
            )

    loss = (
        total_loss
        / len(train_loader)
    )

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

    return (
        loss,
        accuracy,
        macro_f1,
    )


# ============================================================
# Validation
# ============================================================

def validate():

    model.eval()

    total_loss = 0.0

    all_labels = []

    all_predictions = []

    with torch.no_grad():

        for batch in val_loader:

            images = batch[
                "image"
            ].to(device)

            labels = batch[
                "label"
            ].to(device)

            encoding = tokenizer(
                list(batch["text"]),
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )

            input_ids = encoding[
                "input_ids"
            ].to(device)

            attention_mask = encoding[
                "attention_mask"
            ].to(device)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=images,
            )

            loss = criterion(
                logits,
                labels,
            )

            total_loss += loss.item()

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    loss = (
        total_loss
        / len(val_loader)
    )

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

    return (
        loss,
        accuracy,
        macro_f1,
        all_labels,
        all_predictions,
    )


# ============================================================
# Training loop + Early stopping
# ============================================================

print(
    "\n"
    + "=" * 60
)

print(
    "Starting Multimodal V2 training"
)

print(
    "=" * 60
)


best_macro_f1 = 0.0

epochs_without_improvement = 0

best_epoch = 0


for epoch in range(
    EPOCHS
):

    print(
        f"\n## Epoch "
        f"{epoch + 1}/{EPOCHS}"
    )

    print(
        "-" * 60
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    (
        train_loss,
        train_acc,
        train_f1,
    ) = train_one_epoch()

    print(
        "\nTraining Results:"
    )

    print(
        f"Loss:     "
        f"{train_loss:.4f}"
    )

    print(
        f"Accuracy: "
        f"{train_acc:.4f}"
    )

    print(
        f"Macro-F1: "
        f"{train_f1:.4f}"
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print(
        "\nRunning validation..."
    )

    (
        val_loss,
        val_acc,
        val_f1,
        val_labels,
        val_predictions,
    ) = validate()

    print(
        "\nValidation Results:"
    )

    print(
        f"Loss:     "
        f"{val_loss:.4f}"
    )

    print(
        f"Accuracy: "
        f"{val_acc:.4f}"
    )

    print(
        f"Macro-F1: "
        f"{val_f1:.4f}"
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            val_labels,
            val_predictions,
            labels=[
                0,
                1,
                2,
            ],
            target_names=[
                "negative",
                "neutral",
                "positive",
            ],
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if val_f1 > best_macro_f1:

        best_macro_f1 = val_f1

        best_epoch = (
            epoch + 1
        )

        epochs_without_improvement = 0

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": (
                    model.state_dict()
                ),
                "optimizer_state_dict": (
                    optimizer.state_dict()
                ),
                "val_macro_f1": val_f1,
                "val_accuracy": val_acc,
            },
            MODEL_SAVE_PATH,
        )

        print(
            "\n✓ Best V2 model saved!"
        )

        print(
            f"Macro-F1: "
            f"{val_f1:.4f}"
        )

    else:

        epochs_without_improvement += 1

        print(
            "\nNo improvement."
        )

        print(
            f"Patience: "
            f"{epochs_without_improvement}/"
            f"{PATIENCE}"
        )

    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    if (
        epochs_without_improvement
        >= PATIENCE
    ):

        print(
            "\nEarly stopping triggered."
        )

        break


# ============================================================
# Finished
# ============================================================

print(
    "\n"
    + "=" * 60
)

print(
    "Multimodal V2 training completed"
)

print(
    "=" * 60
)

print(
    f"Best Epoch: "
    f"{best_epoch}"
)

print(
    f"Best Validation Macro-F1: "
    f"{best_macro_f1:.4f}"
)

print(
    f"Best model: "
    f"{MODEL_SAVE_PATH}"
)