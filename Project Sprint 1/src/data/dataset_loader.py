from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Paths
PROJECT_DIR = Path(__file__).resolve().parents[2]

DATASET_DIR = PROJECT_DIR / "Dataset" / "processed"


# Configuration
IMAGE_SIZE = 128
BATCH_SIZE = 32
NUM_WORKERS = 0


# Image transformations
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# Dataset and DataLoader creation
def create_dataloaders():

    train_dataset = datasets.ImageFolder(
        DATASET_DIR / "train",
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        DATASET_DIR / "val",
        transform=eval_transform
    )

    test_dataset = datasets.ImageFolder(
        DATASET_DIR / "test",
        transform=eval_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader
    )


# Test the data pipeline
if __name__ == "__main__":

    (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader
    ) = create_dataloaders()

    print("=" * 60)
    print("PYTORCH DATA PIPELINE TEST")
    print("=" * 60)

    print(f"\nDataset directory: {DATASET_DIR}")

    print(f"\nTraining images   : {len(train_dataset)}")
    print(f"Validation images : {len(val_dataset)}")
    print(f"Testing images    : {len(test_dataset)}")

    print("\nClass mapping:")
    print(train_dataset.class_to_idx)

    # Get one training batch
    images, labels = next(iter(train_loader))

    print("\nFirst training batch:")
    print(f"Image tensor shape : {images.shape}")
    print(f"Label tensor shape : {labels.shape}")

    print("\nImage tensor information:")
    print(f"Data type          : {images.dtype}")
    print(f"Minimum value      : {images.min().item():.4f}")
    print(f"Maximum value      : {images.max().item():.4f}")

    print("\n" + "=" * 60)
    print("DATA PIPELINE TEST COMPLETE")
    print("=" * 60)