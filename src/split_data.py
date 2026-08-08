import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path


INPUT_PATH = Path("data/raw/memotion_dataset_7k/labels.csv")
OUTPUT_DIR = Path("data/processed")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Load dataset
df = pd.read_csv(INPUT_PATH)


# Map original 5 sentiment classes to 3 classes
sentiment_map = {
    "very_positive": "positive",
    "positive": "positive",
    "neutral": "neutral",
    "negative": "negative",
    "very_negative": "negative",
}

df["sentiment"] = df["overall_sentiment"].map(sentiment_map)


# Remove rows with missing sentiment labels
df = df.dropna(subset=["sentiment"])


# Stratified train-validation split
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["sentiment"],
)


# Save splits
train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
val_df.to_csv(OUTPUT_DIR / "val.csv", index=False)


print(f"Total samples: {len(df)}")
print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")

print("\nTraining distribution:")
print(train_df["sentiment"].value_counts(normalize=True))

print("\nValidation distribution:")
print(val_df["sentiment"].value_counts(normalize=True))