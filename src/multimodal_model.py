import torch
import torch.nn as nn

from transformers import AutoModel
from torchvision.models import resnet18, ResNet18_Weights


MODEL_NAME = "distilbert-base-uncased"


class MultimodalSentimentModelV4(nn.Module):

    def __init__(
        self,
        text_model_name=MODEL_NAME,
        num_classes=3,
        dropout=0.3,
    ):
        super().__init__()

        # Text encoder: DistilBERT
        self.text_encoder = AutoModel.from_pretrained(
            text_model_name
        )

        text_dim = self.text_encoder.config.hidden_size

        # Image encoder: ResNet18
        self.image_encoder = resnet18(
            weights=ResNet18_Weights.DEFAULT
        )

        image_dim = self.image_encoder.fc.in_features

        # Remove original ResNet classifier
        self.image_encoder.fc = nn.Identity()

        # Text projection: 768 -> 256
        self.text_projection = nn.Linear(
            text_dim,
            256,
        )

        # Image projection: 512 -> 256
        self.image_projection = nn.Linear(
            image_dim,
            256,
        )

        # Fusion classifier: 512 -> 256 -> 3
        self.classifier = nn.Sequential(
            nn.Linear(
                512,
                256,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
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

        # Text features
        text_output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        text_features = (
            text_output.last_hidden_state[:, 0, :]
        )

        text_features = self.text_projection(
            text_features
        )

        # Image features
        image_features = self.image_encoder(
            images
        )

        image_features = self.image_projection(
            image_features
        )

        # Combine text + image
        fused = torch.cat(
            [
                text_features,
                image_features,
            ],
            dim=1,
        )

        # Sentiment prediction
        logits = self.classifier(
            fused
        )

        return logits