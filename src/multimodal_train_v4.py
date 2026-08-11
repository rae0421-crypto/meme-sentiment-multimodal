from pathlib import Path
import random

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from transformers import AutoTokenizer, AutoModel
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

from dataset import MemotionDataset


# ============================================================
# Reproducibility
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# Paths
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

BEST_MODEL_PATH = (
    MODEL_DIR
    / "best_multimodal_v4.pt"
)


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "distilbert-base-uncased"

NUM_CLASSES = 3

MAX_LENGTH = 64

BATCH_SIZE = 16

NUM_EPOCHS = 8

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

PATIENCE = 2

FOCAL_GAMMA = 2.0

DROPOUT = 0.3

LABEL_NAMES = [
    "negative",
    "neutral",
    "positive",
]


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
        f"Train CSV not found: {TRAIN_CSV}"
    )

if not VAL_CSV.exists():
    raise FileNotFoundError(
        f"Val CSV not found: {VAL_CSV}"
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
# Remove known corrupted image from CSV
# ============================================================

CORRUPTED_IMAGE = "image_5119.png"

for csv_path in [
    TRAIN_CSV,
    VAL_CSV,
]:

    df = pd.read_csv(csv_path)

    if (
        CORRUPTED_IMAGE
        in df["image_name"].values
    ):

        df = df[
            df["image_name"]
            != CORRUPTED_IMAGE
        ]

        df.to_csv(
            csv_path,
            index=False,
        )

        print(
            f"Removed corrupted image from {csv_path.name}"
        )


# ============================================================
# Tokenizer
# ============================================================

print(
    "\nLoading tokenizer..."
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
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
# Datasets
# ============================================================

print(
    "\nLoading datasets..."
)

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
    f"Training samples:   {len(train_dataset)}"
)

print(
    f"Validation samples: {len(val_dataset)}"
)


# ============================================================
# Class distribution
# ============================================================

print(
    "\nChecking class distribution..."
)

train_labels = (
    train_dataset.df["label_id"]
    .astype(int)
    .values
)

class_counts = np.bincount(
    train_labels,
    minlength=NUM_CLASSES,
)

for i, count in enumerate(
    class_counts
):

    print(
        f"{i} ({LABEL_NAMES[i]}): {count}"
    )


# ============================================================
# Class weights
#
# We use sqrt inverse frequency rather than
# aggressive inverse frequency.
# ============================================================

class_weights = (
    len(train_labels)
    /
    (
        NUM_CLASSES
        * class_counts
    )
)

class_weights = np.sqrt(
    class_weights
)

# Normalize around 1
class_weights = (
    class_weights
    /
    class_weights.mean()
)

class_weights_tensor = torch.tensor(
    class_weights,
    dtype=torch.float32,
)

print(
    "\nClass weights:"
)

for i, weight in enumerate(
    class_weights
):

    print(
        f"{LABEL_NAMES[i]}: {weight:.4f}"
    )


# ============================================================
# WeightedRandomSampler
#
# sqrt inverse frequency:
# negative gets more exposure,
# but not extreme oversampling.
# ============================================================

class_sample_weights = (
    1.0
    /
    np.sqrt(class_counts)
)

sample_weights = np.array(
    [
        class_sample_weights[label]
        for label in train_labels
    ]
)

sampler = WeightedRandomSampler(
    weights=torch.DoubleTensor(
        sample_weights
    ),

    num_samples=len(
        sample_weights
    ),

    replacement=True,
)


# ============================================================
# DataLoaders
# ============================================================

train_loader = DataLoader(
    train_dataset,

    batch_size=BATCH_SIZE,

    sampler=sampler,

    num_workers=0,

    pin_memory=torch.cuda.is_available(),
)

val_loader = DataLoader(
    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0,

    pin_memory=torch.cuda.is_available(),
)


# ============================================================
# V1-compatible multimodal model
#
# This EXACT structure matches the V1 checkpoint:
#
# DistilBERT
#     ↓
# 768 -> 256
#
# ResNet18
#     ↓
# 512 -> 256
#
# concat
#     ↓
# 512
#
# classifier
# 512 -> 256 -> 3
# ============================================================

class MultimodalSentimentModelV4(
    nn.Module
):

    def __init__(
        self,
        text_model_name=MODEL_NAME,
        num_classes=3,
        dropout=0.3,
    ):

        super().__init__()

        # ----------------------------------------------------
        # Text encoder
        # ----------------------------------------------------

        self.text_encoder = AutoModel.from_pretrained(
            text_model_name
        )

        text_dim = (
            self.text_encoder.config.hidden_size
        )

        # ----------------------------------------------------
        # Image encoder
        # ----------------------------------------------------

        self.image_encoder = resnet18(
            weights=ResNet18_Weights.DEFAULT
        )

        image_dim = (
            self.image_encoder.fc.in_features
        )

        self.image_encoder.fc = nn.Identity()

        # ----------------------------------------------------
        # Projection
        # ----------------------------------------------------

        self.text_projection = nn.Linear(
            text_dim,
            256,
        )

        self.image_projection = nn.Linear(
            image_dim,
            256,
        )

        # ----------------------------------------------------
        # Classifier
        # ----------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Linear(
                512,
                256,
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                256,
                num_classes,
            ),
        )


    def forward(
        self,
        input_ids,
        attention_mask,
        images,
    ):

        # ----------------------------------------------------
        # Text
        # ----------------------------------------------------

        text_output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        text_features = (
            text_output
            .last_hidden_state[:, 0, :]
        )

        text_features = (
            self.text_projection(
                text_features
            )
        )

        # ----------------------------------------------------
        # Image
        # ----------------------------------------------------

        image_features = (
            self.image_encoder(
                images
            )
        )

        image_features = (
            self.image_projection(
                image_features
            )
        )

        # ----------------------------------------------------
        # Fusion
        # ----------------------------------------------------

        fused = torch.cat(
            [
                text_features,
                image_features,
            ],
            dim=1,
        )

        # ----------------------------------------------------
        # Classifier
        # ----------------------------------------------------

        logits = self.classifier(
            fused
        )

        return logits


# ============================================================
# Freeze ResNet
#
# We don't want the image encoder to overfit
# on only ~5.6k training examples.
# ============================================================

print(
    "\nLoading multimodal V4 model..."
)

model = MultimodalSentimentModelV4(
    text_model_name=MODEL_NAME,
    num_classes=NUM_CLASSES,
    dropout=DROPOUT,
)


for param in (
    model.image_encoder.parameters()
):

    param.requires_grad = False


# ============================================================
# Trainable parameters
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
    f"Trainable parameters: {trainable_params:,}"
)

print(
    f"Total parameters:     {total_params:,}"
)


model = model.to(device)


# ============================================================
# Focal Loss
# ============================================================

class FocalLoss(nn.Module):

    def __init__(
        self,
        alpha=None,
        gamma=2.0,
    ):

        super().__init__()

        self.alpha = alpha
        self.gamma = gamma


    def forward(
        self,
        logits,
        targets,
    ):

        log_probs = F.log_softmax(
            logits,
            dim=1,
        )

        probs = torch.exp(
            log_probs
        )

        target_log_probs = (
            log_probs
            .gather(
                1,
                targets.unsqueeze(1),
            )
            .squeeze(1)
        )

        target_probs = (
            probs
            .gather(
                1,
                targets.unsqueeze(1),
            )
            .squeeze(1)
        )

        focal_factor = (
            1.0 - target_probs
        ) ** self.gamma

        loss = (
            -focal_factor
            * target_log_probs
        )

        if self.alpha is not None:

            alpha = self.alpha.to(
                logits.device
            )

            alpha_t = alpha[
                targets
            ]

            loss = (
                alpha_t
                * loss
            )

        return loss.mean()


criterion = FocalLoss(
    alpha=class_weights_tensor,
    gamma=FOCAL_GAMMA,
)


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.AdamW(
    [
        {
            "params":
                model.text_encoder.parameters(),
            "lr":
                LEARNING_RATE * 0.3,
        },

        {
            "params":
                model.text_projection.parameters(),
            "lr":
                LEARNING_RATE,
        },

        {
            "params":
                model.image_projection.parameters(),
            "lr":
                LEARNING_RATE,
        },

        {
            "params":
                model.classifier.parameters(),
            "lr":
                LEARNING_RATE,
        },
    ],

    weight_decay=WEIGHT_DECAY,
)


# ============================================================
# Scheduler
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=1,
)


# ============================================================
# Helper: tokenize
# ============================================================

def tokenize_batch(texts):

    # Make absolutely sure every item is a string.
    texts = [
        "" if text is None
        else str(text)
        for text in texts
    ]

    return tokenizer(
        texts,

        padding=True,

        truncation=True,

        max_length=MAX_LENGTH,

        return_tensors="pt",
    )


# ============================================================
# Train one epoch
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
):

    model.train()

    total_loss = 0.0

    all_preds = []

    all_labels = []

    total_batches = len(loader)


    for batch_idx, batch in enumerate(
        loader,
        start=1,
    ):

        images = batch[
            "image"
        ].to(device)

        labels = batch[
            "label"
        ].to(device)


        texts = batch[
            "text"
        ]


        encoding = tokenize_batch(
            texts
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


        total_loss += (
            loss.item()
        )


        preds = torch.argmax(
            logits,
            dim=1,
        )


        all_preds.extend(
            preds.detach()
            .cpu()
            .numpy()
        )

        all_labels.extend(
            labels.detach()
            .cpu()
            .numpy()
        )


        if (
            batch_idx % 50 == 0
            or batch_idx == total_batches
        ):

            print(
                f"Batch {batch_idx}/{total_batches} "
                f"- Loss: {loss.item():.4f}"
            )


    avg_loss = (
        total_loss
        / total_batches
    )

    accuracy = accuracy_score(
        all_labels,
        all_preds,
    )

    macro_f1 = f1_score(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0,
    )

    return (
        avg_loss,
        accuracy,
        macro_f1,
    )


# ============================================================
# Validation
# ============================================================

def validate(
    model,
    loader,
    criterion,
):

    model.eval()

    total_loss = 0.0

    all_preds = []

    all_labels = []


    with torch.no_grad():

        for batch in loader:

            images = batch[
                "image"
            ].to(device)

            labels = batch[
                "label"
            ].to(device)

            texts = batch[
                "text"
            ]


            encoding = tokenize_batch(
                texts
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


            total_loss += (
                loss.item()
            )


            preds = torch.argmax(
                logits,
                dim=1,
            )


            all_preds.extend(
                preds.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )


    avg_loss = (
        total_loss
        / len(loader)
    )

    accuracy = accuracy_score(
        all_labels,
        all_preds,
    )

    macro_f1 = f1_score(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0,
    )

    return (
        avg_loss,
        accuracy,
        macro_f1,
        all_labels,
        all_preds,
    )


# ============================================================
# Training loop
# ============================================================

best_macro_f1 = -1.0

best_epoch = 0

patience_counter = 0


print(
    "\nStarting V4 training..."
)

print(
    "Strategy:"
)

print(
    "  - V1 architecture"
)

print(
    "  - sqrt WeightedRandomSampler"
)

print(
    "  - Focal Loss"
)

print(
    "  - Frozen ResNet18"
)

print(
    "  - Macro-F1 early stopping"
)


for epoch in range(
    1,
    NUM_EPOCHS + 1,
):

    print(
        "\n"
        + "=" * 60
    )

    print(
        f"Epoch {epoch}/{NUM_EPOCHS}"
    )

    print(
        "=" * 60
    )


    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print(
        "\nTraining..."
    )

    train_loss, train_acc, train_f1 = (
        train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
        )
    )


    print(
        "\nTraining Results:"
    )

    print(
        f"Loss:     {train_loss:.4f}"
    )

    print(
        f"Accuracy: {train_acc:.4f}"
    )

    print(
        f"Macro-F1: {train_f1:.4f}"
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
        val_preds,
    ) = validate(
        model,
        val_loader,
        criterion,
    )


    print(
        "\nValidation Results:"
    )

    print(
        f"Loss:     {val_loss:.4f}"
    )

    print(
        f"Accuracy: {val_acc:.4f}"
    )

    print(
        f"Macro-F1: {val_f1:.4f}"
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

            val_preds,

            labels=[
                0,
                1,
                2,
            ],

            target_names=LABEL_NAMES,

            zero_division=0,
        )
    )


    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler.step(
        val_f1
    )


    # --------------------------------------------------------
    # Save best
    # --------------------------------------------------------

    if val_f1 > best_macro_f1:

        best_macro_f1 = val_f1

        best_epoch = epoch

        patience_counter = 0


        torch.save(
            {
                "epoch":
                    epoch,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "scheduler_state_dict":
                    scheduler.state_dict(),

                "val_macro_f1":
                    val_f1,

                "val_accuracy":
                    val_acc,

                "class_weights":
                    class_weights_tensor,

                "config":
                    {
                        "model_name":
                            MODEL_NAME,

                        "max_length":
                            MAX_LENGTH,

                        "num_classes":
                            NUM_CLASSES,

                        "focal_gamma":
                            FOCAL_GAMMA,

                        "batch_size":
                            BATCH_SIZE,

                        "learning_rate":
                            LEARNING_RATE,
                    },
            },

            BEST_MODEL_PATH,
        )


        print(
            "\n✓ Best V4 model saved!"
        )

        print(
            f"Macro-F1: {val_f1:.4f}"
        )

        print(
            f"Path: {BEST_MODEL_PATH}"
        )

    else:

        patience_counter += 1

        print(
            "\nNo improvement."
        )

        print(
            f"Patience: "
            f"{patience_counter}/{PATIENCE}"
        )


    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    if patience_counter >= PATIENCE:

        print(
            "\nEarly stopping triggered."
        )

        break


# ============================================================
# Final summary
# ============================================================

print(
    "\n"
    + "=" * 60
)

print(
    "V4 TRAINING COMPLETE"
)

print(
    "=" * 60
)

print(
    f"Best Epoch: {best_epoch}"
)

print(
    f"Best Validation Macro-F1: "
    f"{best_macro_f1:.4f}"
)

print(
    f"Best model: {BEST_MODEL_PATH}"
)

print(
    "\nCompare with previous versions:"
)

print(
    "V1: Macro-F1 = 0.3511"
)

print(
    "V2: Macro-F1 = 0.2486"
)

print(
    "V3: Macro-F1 = 0.2820"
)

print(
    f"V4: Macro-F1 = {best_macro_f1:.4f}"
)