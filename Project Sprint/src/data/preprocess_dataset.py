from pathlib import Path
from collections import Counter
from shutil import copy2

from sklearn.model_selection import train_test_split


# Paths
PROJECT_DIR = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_DIR / "Dataset" / "raw"
PROCESSED_DIR = PROJECT_DIR / "Dataset" / "processed"

# Reproducibility
RANDOM_SEED = 42

# Required split
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# Helper functions
def get_images(class_dir):
    """Return all supported image files in a class directory."""
    return sorted(
        [
            path
            for path in class_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


def copy_images(image_paths, destination_dir):
    """Copy images into the specified split directory."""
    destination_dir.mkdir(parents=True, exist_ok=True)

    for image_path in image_paths:
        copy2(image_path, destination_dir / image_path.name)


# Main preprocessing function
def preprocess_dataset():

    print("=" * 70)
    print("TRASHNET DATASET PREPROCESSING")
    print("=" * 70)

    if not RAW_DIR.exists():
        print(f"ERROR: Raw dataset not found at:\n{RAW_DIR}")
        return

    # Find class folders
    class_dirs = sorted(
        folder
        for folder in RAW_DIR.iterdir()
        if folder.is_dir()
    )

    print(f"\nRaw dataset: {RAW_DIR}")
    print(f"Number of classes: {len(class_dirs)}")

    # Create output directories
    for split in ["train", "val", "test"]:
        for class_dir in class_dirs:
            (PROCESSED_DIR / split / class_dir.name).mkdir(
                parents=True,
                exist_ok=True
            )

    total_counts = Counter()
    split_counts = {
        "train": Counter(),
        "val": Counter(),
        "test": Counter()
    }


    # Process each class separately
    for class_dir in class_dirs:

        class_name = class_dir.name
        images = get_images(class_dir)

        total_counts[class_name] = len(images)

        # First split:
        # 70% training
        # 30% temporary
        train_images, temp_images = train_test_split(
            images,
            test_size=(1 - TRAIN_RATIO),
            random_state=RANDOM_SEED,
            shuffle=True
        )

        # Second split:
        # Divide remaining 30% equally:
        # 15% validation
        # 15% testing
        val_images, test_images = train_test_split(
            temp_images,
            test_size=0.50,
            random_state=RANDOM_SEED,
            shuffle=True
        )

        # Copy files
        copy_images(
            train_images,
            PROCESSED_DIR / "train" / class_name
        )

        copy_images(
            val_images,
            PROCESSED_DIR / "val" / class_name
        )

        copy_images(
            test_images,
            PROCESSED_DIR / "test" / class_name
        )

        split_counts["train"][class_name] = len(train_images)
        split_counts["val"][class_name] = len(val_images)
        split_counts["test"][class_name] = len(test_images)

    # Print results
    print("\n" + "-" * 70)
    print("ORIGINAL DATASET")
    print("-" * 70)

    for class_name, count in total_counts.items():
        print(f"{class_name:12} : {count}")

    print(f"\nTotal images: {sum(total_counts.values())}")

    for split in ["train", "val", "test"]:

        print("\n" + "-" * 70)
        print(f"{split.upper()} SET")
        print("-" * 70)

        total_split = 0

        for class_name in total_counts:
            count = split_counts[split][class_name]
            total_split += count
            print(f"{class_name:12} : {count}")

        print(f"Total {split} images: {total_split}")

    # Final verification
    total_train = sum(split_counts["train"].values())
    total_val = sum(split_counts["val"].values())
    total_test = sum(split_counts["test"].values())

    total_processed = total_train + total_val + total_test

    print("\n" + "=" * 70)
    print("FINAL DATASET SPLIT")
    print("=" * 70)

    print(f"Training   : {total_train} images")
    print(f"Validation : {total_val} images")
    print(f"Testing    : {total_test} images")
    print(f"Total      : {total_processed} images")

    print("\nExpected ratio:")
    print("Training   : 70%")
    print("Validation : 15%")
    print("Testing    : 15%")

    print(f"\nRandom seed: {RANDOM_SEED}")

    if total_processed == sum(total_counts.values()):
        print("\nSUCCESS: All valid images were assigned to a dataset split.")
    else:
        print("\nWARNING: Image count mismatch detected.")

    print("\nProcessed dataset saved to:")
    print(PROCESSED_DIR)

    print("\n" + "=" * 70)
    print("DATASET PREPROCESSING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    preprocess_dataset()