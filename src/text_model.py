import torch
import torch.nn as nn
from transformers import AutoModel


class TextSentimentModel(nn.Module):
    def __init__(
        self,
        model_name="bert-base-uncased",
        num_classes=3,
        dropout=0.3,
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)

        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(
            hidden_size,
            num_classes,
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Use the [CLS] token representation
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        x = self.dropout(cls_embedding)

        logits = self.classifier(x)

        return logits