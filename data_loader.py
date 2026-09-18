from pathlib import Path
import csv

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms


# ==================================================
# PATHS
# ==================================================

PROJECT_PATH = Path(__file__).parent
MANIFEST_PATH = PROJECT_PATH / "dataset_split.csv"


# ==================================================
# TRANSFORMS
# ==================================================

# Training:
# Mild geometric augmentation only.
# We avoid strong color changes because conjunctival
# color may contain useful diagnostic information.

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(
        224,
        scale=(0.90, 1.0)
    ),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Validation and test:
# NO augmentation.

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==================================================
# LABELS
# ==================================================

LABEL_MAP = {
    "Healthy": 0,
    "Diabetes": 1
}


# ==================================================
# DATASET
# ==================================================

class ConjunctivalDataset(Dataset):

    def __init__(self, csv_file, split, transform=None):

        self.transform = transform
        self.samples = []

        with open(csv_file, "r", encoding="utf-8") as file:

            reader = csv.DictReader(file)

            for row in reader:

                if row["split"] == split:
                    self.samples.append(row)

        print(
            f"{split.capitalize()} samples: "
            f"{len(self.samples)}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        row = self.samples[index]

        image_path = Path(row["image_path"])
        label = LABEL_MAP[row["label"]]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(
            label,
            dtype=torch.long
        )


# ==================================================
# DATASETS
# ==================================================

train_dataset = ConjunctivalDataset(
    MANIFEST_PATH,
    split="train",
    transform=train_transform
)

val_dataset = ConjunctivalDataset(
    MANIFEST_PATH,
    split="val",
    transform=eval_transform
)

test_dataset = ConjunctivalDataset(
    MANIFEST_PATH,
    split="test",
    transform=eval_transform
)


# ==================================================
# DATALOADERS
# ==================================================

BATCH_SIZE = 16

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ==================================================
# BASIC CHECK
# ==================================================

print("\n" + "=" * 55)
print("DATA LOADERS READY")
print("=" * 55)

images, labels = next(iter(train_loader))

print(f"Train batch shape : {images.shape}")
print(f"Labels shape      : {labels.shape}")
print("=" * 55)