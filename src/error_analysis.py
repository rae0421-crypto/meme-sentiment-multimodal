from pathlib import Path

import torch
import torch.nn as nn
import pandas as pd

from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModel

from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

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

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "best_multimodal_model.pt"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "error_analysis.csv"
)


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "distilbert-base-uncased"

BATCH_SIZE = 16

MAX_LENGTH = 64

NUM_CLASSES = 3

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
# V1 MODEL
# ============================================================

class V1MultimodalModel(nn.Module):

    def __init__(
        self,
        text_model_name="distilbert-base-uncased",
        num_classes=3,
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
        #
        # This matches the V1 checkpoint:
        #
        # text_projection.weight
        # text_projection.bias
        #
        # image_projection.weight
        # image_projection.bias
        #
        # classifier.0
        # classifier.3
        # ----------------------------------------------------

        self.classifier = nn.Sequential(
            nn.Linear(
                256 * 2,
                256,
            ),

            nn.ReLU(),

            nn.Dropout(
                0.3
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

        # CLS representation
        text_features = (
            text_output.last_hidden_state[:, 0, :]
        )

        text_features = self.text_projection(
            text_features
        )

        # ----------------------------------------------------
        # Image
        # ----------------------------------------------------

        image_features = self.image_encoder(
            images
        )

        image_features = self.image_projection(
            image_features
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
        # Classification
        # ----------------------------------------------------

        logits = self.classifier(
            fused
        )

        return logits


# ============================================================
# Transform
# ============================================================

transform = transforms.Compose(
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
# Tokenizer
# ============================================================

print(
    "\nLoading tokenizer..."
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ============================================================
# Dataset
# ============================================================

print(
    "\nLoading validation dataset..."
)


# IMPORTANT:
# We import your existing dataset.py here.
#
from dataset import MemotionDataset


dataset = MemotionDataset(
    csv_path=VAL_CSV,
    image_dir=IMAGE_DIR,
    tokenizer=tokenizer,
    transform=transform,
)


loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


print(
    f"Validation samples: {len(dataset)}"
)


# ============================================================
# Model
# ============================================================

print(
    "\nLoading V1 model..."
)


model = V1MultimodalModel(
    text_model_name=MODEL_NAME,
    num_classes=NUM_CLASSES,
)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
)


print(
    "\nCheckpoint keys:"
)

if isinstance(
    checkpoint,
    dict,
):

    print(
        checkpoint.keys()
    )


# ------------------------------------------------------------
# Load checkpoint
# ------------------------------------------------------------

if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):

    state_dict = checkpoint[
        "model_state_dict"
    ]

else:

    state_dict = checkpoint


model.load_state_dict(
    state_dict
)


model = model.to(device)

model.eval()


print(
    "✓ V1 model loaded successfully."
)


# ============================================================
# Prediction
# ============================================================

all_results = []

all_true = []

all_pred = []

print(
    "\nRunning predictions..."
)


sample_index = 0


with torch.no_grad():

    for batch in loader:

        images = batch[
            "image"
        ].to(device)

        labels = batch[
            "label"
        ].to(device)

        texts = list(
            batch["text"]
        )

        encoding = tokenizer(
            texts,

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


        # ----------------------------------------------------
        # Save each sample
        # ----------------------------------------------------

        for i in range(
            len(labels)
        ):

            true_label = (
                labels[i]
                .item()
            )

            pred_label = (
                predictions[i]
                .item()
            )

            confidence = (
                confidences[i]
                .item()
            )

            row = dataset.df.iloc[
                sample_index
            ]

            image_name = row[
                "image_name"
            ]

            text = texts[i]


            all_true.append(
                true_label
            )

            all_pred.append(
                pred_label
            )


            all_results.append(
                {
                    "index":
                        sample_index,

                    "image_name":
                        image_name,

                    "text":
                        text,

                    "true_label":
                        LABEL_NAMES[
                            true_label
                        ],

                    "predicted_label":
                        LABEL_NAMES[
                            pred_label
                        ],

                    "confidence":
                        confidence,

                    "correct":
                        true_label
                        == pred_label,
                }
            )


            sample_index += 1


# ============================================================
# Classification Report
# ============================================================

print(
    "\n"
    + "=" * 60
)

print(
    "Classification Report"
)

print(
    "=" * 60
)


print(
    classification_report(
        all_true,

        all_pred,

        labels=[
            0,
            1,
            2,
        ],

        target_names=LABEL_NAMES,

        zero_division=0,
    )
)


# ============================================================
# Confusion Matrix
# ============================================================

print(
    "\n"
    + "=" * 60
)

print(
    "Confusion Matrix"
)

print(
    "=" * 60
)


cm = confusion_matrix(
    all_true,

    all_pred,

    labels=[
        0,
        1,
        2,
    ],
)


cm_df = pd.DataFrame(
    cm,

    index=[
        "true_negative",
        "true_neutral",
        "true_positive",
    ],

    columns=[
        "pred_negative",
        "pred_neutral",
        "pred_positive",
    ],
)


print(
    cm_df
)


# ============================================================
# Save CSV
# ============================================================

results_df = pd.DataFrame(
    all_results
)


results_df.to_csv(
    OUTPUT_PATH,

    index=False,
)


print(
    "\n✓ Error analysis saved:"
)

print(
    OUTPUT_PATH
)


# ============================================================
# Error groups
# ============================================================

print(
    "\n"
    + "=" * 60
)

print(
    "Most Important Error Groups"
)

print(
    "=" * 60
)


# ============================================================
# Negative → Positive
# ============================================================

group = results_df[
    (
        results_df["true_label"]
        == "negative"
    )
    &
    (
        results_df["predicted_label"]
        == "positive"
    )
].sort_values(
    "confidence",
    ascending=False,
)


print(
    "\nNegative → Positive:"
)


if len(group) > 0:

    print(
        group[
            [
                "image_name",
                "text",
                "confidence",
            ]
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

else:

    print(
        "No samples."
    )


# ============================================================
# Negative → Neutral
# ============================================================

group = results_df[
    (
        results_df["true_label"]
        == "negative"
    )
    &
    (
        results_df["predicted_label"]
        == "neutral"
    )
].sort_values(
    "confidence",
    ascending=False,
)


print(
    "\nNegative → Neutral:"
)


if len(group) > 0:

    print(
        group[
            [
                "image_name",
                "text",
                "confidence",
            ]
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

else:

    print(
        "No samples."
    )


# ============================================================
# Neutral → Positive
# ============================================================

group = results_df[
    (
        results_df["true_label"]
        == "neutral"
    )
    &
    (
        results_df["predicted_label"]
        == "positive"
    )
].sort_values(
    "confidence",
    ascending=False,
)


print(
    "\nNeutral → Positive:"
)


if len(group) > 0:

    print(
        group[
            [
                "image_name",
                "text",
                "confidence",
            ]
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

else:

    print(
        "No samples."
    )


# ============================================================
# Positive → Neutral
# ============================================================

group = results_df[
    (
        results_df["true_label"]
        == "positive"
    )
    &
    (
        results_df["predicted_label"]
        == "neutral"
    )
].sort_values(
    "confidence",
    ascending=False,
)


print(
    "\nPositive → Neutral:"
)


if len(group) > 0:

    print(
        group[
            [
                "image_name",
                "text",
                "confidence",
            ]
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

else:

    print(
        "No samples."
    )


# ============================================================
# High confidence errors
# ============================================================

high_conf_errors = results_df[
    results_df["correct"] == False
].sort_values(
    "confidence",
    ascending=False,
)


print(
    "\nHigh-confidence errors:"
)


print(
    high_conf_errors[
        [
            "image_name",
            "text",
            "true_label",
            "predicted_label",
            "confidence",
        ]
    ]
    .head(20)
    .to_string(
        index=False
    )
)


print(
    "\n✓ Analysis complete."
)