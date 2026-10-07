from pathlib import Path
from PIL import Image
from collections import Counter


# Path to the raw TrashNet dataset
DATASET_DIR = Path(__file__).resolve().parents[2] / "Dataset" / "raw"

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def inspect_dataset():
    print("=" * 60)
    print("TRASHNET DATASET INSPECTION")
    print("=" * 60)

    if not DATASET_DIR.exists():
        print(f"Dataset directory not found: {DATASET_DIR}")
        return

    class_counts = {}
    image_sizes = Counter()
    file_extensions = Counter()
    invalid_images = []

    # Each folder represents one waste class
    class_folders = sorted(
        folder for folder in DATASET_DIR.iterdir() if folder.is_dir()
    )

    print(f"\nDataset location: {DATASET_DIR}")
    print(f"Number of classes: {len(class_folders)}")

    print("\nClasses found:")
    for folder in class_folders:
        print(f"  - {folder.name}")

    # Inspect each class
    for class_folder in class_folders:
        count = 0

        for image_path in class_folder.iterdir():
            if not image_path.is_file():
                continue

            file_extensions[image_path.suffix.lower()] += 1

            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            try:
                with Image.open(image_path) as image:
                    image.verify()

                with Image.open(image_path) as image:
                    image_sizes[image.size] += 1

                count += 1

            except Exception:
                invalid_images.append(str(image_path))

        class_counts[class_folder.name] = count

    total_images = sum(class_counts.values())

    print("\n" + "-" * 60)
    print("IMAGE COUNT BY CLASS")
    print("-" * 60)

    for class_name, count in class_counts.items():
        print(f"{class_name:12} : {count}")

    print(f"\nTotal valid images: {total_images}")

    print("\n" + "-" * 60)
    print("IMAGE DIMENSIONS")
    print("-" * 60)

    for size, count in image_sizes.most_common():
        print(f"{size}: {count} images")

    print("\n" + "-" * 60)
    print("FILE EXTENSIONS")
    print("-" * 60)

    for extension, count in file_extensions.items():
        print(f"{extension:8} : {count}")

    print("\n" + "-" * 60)
    print("INVALID / CORRUPTED IMAGES")
    print("-" * 60)

    if invalid_images:
        print(f"Found: {len(invalid_images)}")
        for image_path in invalid_images:
            print(f"  - {image_path}")
    else:
        print("None found.")

    print("\n" + "=" * 60)
    print("DATASET INSPECTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    inspect_dataset()