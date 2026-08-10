import torch
import torch.nn as nn
from transformers import AutoModel
from torchvision.models import resnet18, ResNet18_Weights


class MultimodalSentimentModel(nn.Module):

    def __init__(
        self,
        text_model_name="distilbert-base-uncased",
        num_classes=3,
    ):
        super().__init__()

        # =====================================================
        # Text Encoder
        # =====================================================

        self.text_encoder = AutoModel.from_pretrained(
            text_model_name
        )

        text_dim = self.text_encoder.config.hidden_size

        # =====================================================
        # Image Encoder
        # =====================================================

        self.image_encoder = resnet18(
            weights=ResNet18_Weights.DEFAULT
        )

        image_dim = self.image_encoder.fc.in_features

        self.image_encoder.fc = nn.Identity()

        # =====================================================
        # Projection layers
        # =====================================================

        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
        )

        self.image_projection = nn.Sequential(
            nn.Linear(image_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
        )

        # =====================================================
        # Multimodal fusion
        # =====================================================

        fusion_dim = 256 * 3

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.5),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.4),
        )

        # =====================================================
        # Classifier
        # =====================================================

        self.classifier = nn.Linear(
            128,
            num_classes,
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        images,
    ):

        # =====================================================
        # Text
        # =====================================================

        text_output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Mean pooling
        text_features = text_output.last_hidden_state.mean(
            dim=1
        )

        text_features = self.text_projection(
            text_features
        )

        # =====================================================
        # Image
        # =====================================================

        image_features = self.image_encoder(
            images
        )

        image_features = self.image_projection(
            image_features
        )

        # =====================================================
        # Cross-modal interaction
        # =====================================================

        interaction = (
            text_features * image_features
        )

        # =====================================================
        # Fusion
        # =====================================================

        fused_features = torch.cat(
            [
                text_features,
                image_features,
                interaction,
            ],
            dim=1,
        )

        fused_features = self.fusion(
            fused_features
        )

        # =====================================================
        # Classification
        # =====================================================

        logits = self.classifier(
            fused_features
        )

        return logits