"""Evaluate the V4 multimodal checkpoint on a Memotion CSV split."""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from dataset import MemotionDataset
from inference import IMAGE_TRANSFORM, LABEL_NAMES, load_v4_checkpoint


BASE_DIR = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=BASE_DIR / "data/processed/val.csv")
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=BASE_DIR / "data/raw/memotion_dataset_7k/images",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=BASE_DIR / "models/best_multimodal_v4.pt"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path, default=BASE_DIR / "results/v4_test_metrics.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, tokenizer, device, checkpoint = load_v4_checkpoint(args.checkpoint)
    max_length = checkpoint.get("config", {}).get("max_length", 64)
    dataset = MemotionDataset(args.csv, args.image_dir, tokenizer, IMAGE_TRANSFORM)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    labels: list[int] = []
    predictions: list[int] = []
    with torch.inference_mode():
        for batch in loader:
            encoding = tokenizer(
                ["" if pd.isna(text) else str(text) for text in batch["text"]],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            logits = model(
                input_ids=encoding["input_ids"].to(device),
                attention_mask=encoding["attention_mask"].to(device),
                images=batch["image"].to(device),
            )
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            labels.extend(batch["label"].tolist())

    metrics = {
        "checkpoint": str(args.checkpoint),
        "split": str(args.csv),
        "samples": len(labels),
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1, 2]).tolist(),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=[0, 1, 2],
            target_names=LABEL_NAMES,
            output_dict=True,
            zero_division=0,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2))
    print(f"Full metrics saved to {args.output}")


if __name__ == "__main__":
    main()
