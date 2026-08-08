import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import MemotionDataset


dataset = MemotionDataset(
    csv_path=PROJECT_ROOT / "data/raw/memotion_dataset_7k/labels.csv",
    image_dir=PROJECT_ROOT / "data/raw/memotion_dataset_7k/images",
)

print("Dataset size:", len(dataset))

sample = dataset[0]

print("Text:", sample["text"])
print("Label:", sample["label"])
print("Image:", sample["image"])