# Multimodal Meme Sentiment Classification

A PyTorch project that classifies Memotion memes as **negative**, **neutral**, or
**positive**. It compares a DistilBERT text-only baseline with multimodal models
that combine DistilBERT text features and pretrained ResNet18 image features.

## Current experiments

| Model | Main change | Validation macro-F1 |
|---|---|---:|
| V1 | DistilBERT + ResNet18 concatenation | 0.3511 |
| V2 | Frozen encoders / weighted loss experiment | 0.2486 |
| V3 | Interaction-based fusion | 0.2820 |
| V4 | V1 architecture + sqrt sampling + focal loss + frozen ResNet18 | Run-dependent |

V4 saves its best validation macro-F1 and accuracy inside
`models/best_multimodal_v4.pt`. The held-out test result should be reported only
once model selection is complete.

## Project structure

```text
app/streamlit_app.py          Streamlit prediction interface
src/train.py                  Text-only baseline training
src/multimodal_train_v*.py    Multimodal experiments V1-V4
src/evaluate.py               Reproducible V4 evaluation
src/inference.py              Shared V4 loading and prediction code
```

The dataset and trained checkpoints are intentionally not committed. Expected
local paths are:

```text
data/raw/memotion_dataset_7k/images/
data/processed/train.csv
data/processed/val.csv
data/processed/test.csv
models/best_multimodal_v4.pt
```

CSV files must include `image_name`, `text_corrected`, and either the original
`overall_sentiment` field used by `MemotionDataset` or the processed sentiment
fields used by the text baseline.

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/) are recommended.

```bash
git clone https://github.com/rae0421-crypto/meme-sentiment-multimodal.git
cd meme-sentiment-multimodal
uv sync
```

On Windows PowerShell, activate the environment if you want to run commands
without the `uv run` prefix:

```powershell
.venv\Scripts\Activate.ps1
```

## Run training

Text-only baseline:

```bash
uv run python src/train.py
```

Current multimodal experiment:

```bash
uv run python src/multimodal_train_v4.py
```

Training requires the Memotion images and processed train/validation CSV files.
CUDA is strongly recommended; CPU training will be slow.

## Evaluate the final model

Use the held-out test split for the final comparison:

```bash
uv run python src/evaluate.py \
  --csv data/processed/test.csv \
  --checkpoint models/best_multimodal_v4.pt \
  --output results/v4_test_metrics.json
```

On PowerShell, place the command on one line or replace `\` with the PowerShell
continuation character (backtick).

The output includes accuracy, macro-F1, a 3×3 confusion matrix, and per-class
precision/recall/F1. The core research comparison is valid only when the
text-only and multimodal models use the exact same split and label mapping.

## Run the Streamlit app

```bash
uv run streamlit run app/streamlit_app.py
```

Open the local URL printed by Streamlit, upload a meme, paste its text, and click
**Predict sentiment**. The app shows the predicted label, confidence, and all
three class probabilities.

## Expected result

The desired result is not a particular fixed score. Evidence for the project
hypothesis requires V4 test macro-F1 to exceed the text-only test macro-F1 on the
same held-out examples. Report accuracy as a secondary metric because macro-F1
is more informative for the imbalanced sentiment classes. Also inspect the
confusion matrix and per-class recall—especially the minority negative class—to
verify that improvement is not caused only by the majority positive class.
