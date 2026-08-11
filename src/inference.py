"""Shared model loading and inference utilities for the V4 checkpoint."""

from pathlib import Path
from typing import Any

from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18
from transformers import AutoModel, AutoTokenizer


LABEL_NAMES = ["negative", "neutral", "positive"]
DEFAULT_MODEL_NAME = "distilbert-base-uncased"
DEFAULT_MAX_LENGTH = 64


class MultimodalSentimentModelV4(nn.Module):
    """Architecture used by ``multimodal_train_v4.py``."""

    def __init__(
        self,
        text_model_name: str = DEFAULT_MODEL_NAME,
        num_classes: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        text_dim = self.text_encoder.config.hidden_size

        self.image_encoder = resnet18(weights=ResNet18_Weights.DEFAULT)
        image_dim = self.image_encoder.fc.in_features
        self.image_encoder.fc = nn.Identity()

        self.text_projection = nn.Linear(text_dim, 256)
        self.image_projection = nn.Linear(image_dim, 256)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        images: torch.Tensor,
    ) -> torch.Tensor:
        text_output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        text_features = self.text_projection(
            text_output.last_hidden_state[:, 0, :]
        )
        image_features = self.image_projection(self.image_encoder(images))
        return self.classifier(torch.cat([text_features, image_features], dim=1))


IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def load_v4_checkpoint(
    checkpoint_path: str | Path,
    device: str | torch.device | None = None,
) -> tuple[MultimodalSentimentModelV4, Any, torch.device, dict[str, Any]]:
    """Load a V4 checkpoint saved by ``multimodal_train_v4.py``."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    resolved_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    checkpoint = torch.load(
        checkpoint_path,
        map_location=resolved_device,
        weights_only=False,
    )
    config = checkpoint.get("config", {})
    model_name = config.get("model_name", DEFAULT_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = MultimodalSentimentModelV4(text_model_name=model_name)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(resolved_device).eval()
    return model, tokenizer, resolved_device, checkpoint


def predict(
    model: MultimodalSentimentModelV4,
    tokenizer: Any,
    image: Image.Image,
    text: str,
    device: torch.device,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> dict[str, Any]:
    """Predict sentiment and return all class probabilities."""
    encoding = tokenizer(
        text or "",
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    image_tensor = IMAGE_TRANSFORM(image.convert("RGB")).unsqueeze(0)

    with torch.inference_mode():
        logits = model(
            input_ids=encoding["input_ids"].to(device),
            attention_mask=encoding["attention_mask"].to(device),
            images=image_tensor.to(device),
        )
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()

    predicted_id = int(torch.tensor(probabilities).argmax().item())
    return {
        "label": LABEL_NAMES[predicted_id],
        "confidence": probabilities[predicted_id],
        "probabilities": dict(zip(LABEL_NAMES, probabilities)),
    }
