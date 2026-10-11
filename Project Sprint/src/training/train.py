from pathlib import Path
import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim


# Allow imports from src/
SRC_DIR = Path(__file__).resolve().parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


from data.dataset_loader import create_dataloaders
from model.baseline_cnn import BaselineCNN


# Configuration
NUM_CLASSES = 6
LEARNING_RATE = 0.001
NUM_EPOCHS = 20

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

PROJECT_DIR = Path(__file__).resolve().parents[2]

CHECKPOINT_DIR = PROJECT_DIR / "models" / "checkpoints"
RESULTS_DIR = PROJECT_DIR / "results"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = CHECKPOINT_DIR / "baseline_cnn_best.pth"
HISTORY_PATH = RESULTS_DIR / "training_history.json"


# Evaluation function
def evaluate(model, data_loader, criterion):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in data_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predictions = torch.argmax(outputs, dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


# Training function
def train():

    print("=" * 70)
    print("TRASHNET BASELINE CNN TRAINING")
    print("=" * 70)

    print(f"\nDevice: {DEVICE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")

    # Load datasets
    (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader
    ) = create_dataloaders()

    print("\nDataset:")
    print(f"Training images   : {len(train_dataset)}")
    print(f"Validation images : {len(val_dataset)}")
    print(f"Testing images    : {len(test_dataset)}")

    print("\nClasses:")
    print(train_dataset.class_to_idx)

    # Create model
    model = BaselineCNN(
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    print("\nModel:")
    print(model)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # Training history
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }

    best_val_accuracy = 0.0


    # Training loop
    for epoch in range(NUM_EPOCHS):

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # Clear gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Backpropagation
            loss.backward()

            # Update weights
            optimizer.step()

            # Statistics
            running_loss += loss.item() * images.size(0)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

        train_loss = running_loss / total
        train_accuracy = correct / total

        # Validation
        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            criterion
        )

        # Save history
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)

        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        print(
            f"Epoch [{epoch + 1:02d}/{NUM_EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )

        # Save best model
        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": val_accuracy,
                    "class_to_idx": train_dataset.class_to_idx
                },
                BEST_MODEL_PATH
            )

            print(
                f"  → Best model saved "
                f"(validation accuracy: {val_accuracy:.4f})"
            )

    # Save training history
    with open(
        HISTORY_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            history,
            file,
            indent=4
        )

    # Load best model
    checkpoint = torch.load(
        BEST_MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # Final test evaluation
    test_loss, test_accuracy = evaluate(
        model,
        test_loader,
        criterion
    )

    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)

    print(f"Best validation accuracy : {best_val_accuracy:.4f}")
    print(f"Test loss                : {test_loss:.4f}")
    print(f"Test accuracy            : {test_accuracy:.4f}")

    print("\nModel saved to:")
    print(BEST_MODEL_PATH)

    print("\nTraining history saved to:")
    print(HISTORY_PATH)

    print("\n" + "=" * 70)
    print("BASELINE CNN TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    train()