
### Smart Waste Image Classification and Monitoring System
### Sprint 1 - Baseline CNN Evaluation
### Evaluates the trained BaselineCNN model on the test dataset.

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

import matplotlib.pyplot as plt


# PROJECT PATHS
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = PROJECT_ROOT / "Dataset" / "processed"
MODEL_PATH = PROJECT_ROOT / "models" / "checkpoints" / "baseline_cnn_best.pth"
RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# CONFIGURATION
IMAGE_SIZE = 128
BATCH_SIZE = 32

CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# BASELINE CNN
class BaselineCNN(nn.Module):
    """
    Baseline CNN used for six-class waste classification.
    """

    def __init__(self, num_classes=6):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# MAIN EVALUATION
def evaluate():

    print("=" * 70)
    print("TRASHNET BASELINE CNN EVALUATION")
    print("=" * 70)

    print(f"\nDevice: {DEVICE}")
    print(f"Test dataset: {DATASET_DIR / 'test'}")
    print(f"Model: {MODEL_PATH}")

    # TRANSFORM
    # Must match the training preprocessing.
    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # LOAD TEST DATASET
    test_dataset = datasets.ImageFolder(
        DATASET_DIR / "test",
        transform=test_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print(f"\nNumber of test images: {len(test_dataset)}")
    print(f"Classes: {test_dataset.classes}")

    # LOAD MODEL
    model = BaselineCNN(num_classes=len(CLASS_NAMES))

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict( 
        checkpoint["model_state_dict"]
    )
    model = model.to(DEVICE)
    model.eval()

    print("\nModel loaded successfully.")

    # RUN INFERENCE
    all_labels = []
    all_predictions = []

    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)

            _, predictions = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    # CALCULATE METRICS
    test_loss = total_loss / len(test_dataset)

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )

    # PRINT RESULTS
    print("\n" + "=" * 70)
    print("OVERALL TEST RESULTS")
    print("=" * 70)

    print(f"Test Loss     : {test_loss:.4f}")
    print(f"Accuracy      : {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Precision     : {precision:.4f}")
    print(f"Recall        : {recall:.4f}")
    print(f"F1-score      : {f1:.4f}")

    # CLASSIFICATION REPORT
    print("\n" + "=" * 70)
    print("PER-CLASS CLASSIFICATION REPORT")
    print("=" * 70)

    report = classification_report(
        all_labels,
        all_predictions,
        target_names=CLASS_NAMES,
        zero_division=0
    )

    print(report)

    # CONFUSION MATRIX
    cm = confusion_matrix(
        all_labels,
        all_predictions
    )

    print("=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    print(cm)

    # SAVE CONFUSION MATRIX IMAGE
    plt.figure(figsize=(8, 6))

    plt.imshow(cm)

    plt.title("Baseline CNN - Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")

    plt.xticks(
        range(len(CLASS_NAMES)),
        CLASS_NAMES,
        rotation=45
    )

    plt.yticks(
        range(len(CLASS_NAMES)),
        CLASS_NAMES
    )

    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    plt.colorbar()
    plt.tight_layout()

    confusion_matrix_path = RESULTS_DIR / "confusion_matrix.png"

    plt.savefig(
        confusion_matrix_path,
        dpi=300
    )

    plt.close()

    # SAVE NUMERICAL RESULTS
    evaluation_results = {
        "model": "BaselineCNN",
        "device": str(DEVICE),
        "test_images": len(test_dataset),
        "test_loss": round(float(test_loss), 6),
        "accuracy": round(float(accuracy), 6),
        "precision_weighted": round(float(precision), 6),
        "recall_weighted": round(float(recall), 6),
        "f1_weighted": round(float(f1), 6),
        "classes": CLASS_NAMES,
        "confusion_matrix": cm.tolist(),
    }

    results_path = RESULTS_DIR / "evaluation_results.json"

    with open(results_path, "w") as file:
        json.dump(
            evaluation_results,
            file,
            indent=4
        )

    # COMPLETION
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print("\nResults saved to:")
    print(results_path)

    print("\nConfusion matrix saved to:")
    print(confusion_matrix_path)

    print("\n" + "=" * 70)


if __name__ == "__main__":
    evaluate()