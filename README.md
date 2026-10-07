# Smart Waste Image Classification and Monitoring System

## Project Overview

The Smart Waste Image Classification and Monitoring System is a computer vision project that classifies a single waste item into one of six predefined waste categories:

- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Trash

The project uses the TrashNet dataset and a convolutional neural network (CNN) implemented using PyTorch. The system is being developed iteratively, beginning with dataset preparation and baseline model development in Sprint 1, followed by model refinement and application integration in later stages.

## Dataset

The project uses the TrashNet dataset.

The dataset contains 2,527 images distributed across six waste categories.

### Dataset Classes

| Class | Number of Images |
|---|---:|
| Cardboard | 403 |
| Glass | 501 |
| Metal | 410 |
| Paper | 594 |
| Plastic | 482 |
| Trash | 137 |
| **Total** | **2,527** |

### Dataset Split

The dataset was divided into:

- Training: 1,765 images (70%)
- Validation: 379 images (15%)
- Testing: 383 images (15%)

The raw dataset and processed image files are excluded from Git because of their size. The preprocessing script can recreate the processed dataset from the raw dataset.

## Sprint 1

Sprint 1 focuses on establishing the machine learning foundation of the project.

### Completed Tasks

- Dataset inspection
- Dataset preprocessing
- Dataset splitting
- PyTorch dataset loading pipeline
- Baseline CNN implementation
- Model input/output verification
- Baseline CNN training
- Test-set evaluation
- Classification report generation
- Confusion matrix generation
- Training history recording
- Git version control setup

## Baseline CNN

The baseline model is implemented using PyTorch.

### Model Configuration

- Framework: PyTorch
- Input size: 128 × 128 RGB
- Number of classes: 6
- Training device: CPU
- Number of epochs: 20
- Learning rate: 0.001

The baseline CNN consists of three convolutional blocks followed by a fully connected classifier.

The model produces six output classes corresponding to the six TrashNet categories.

## Baseline Results

The initial baseline model achieved the following results on the test dataset:

| Metric | Result |
|---|---:|
| Test Loss | 1.0590 |
| Accuracy | 69.45% |
| Precision | 69.75% |
| Recall | 69.45% |
| F1-Score | 69.23% |

These results serve as the baseline for further model improvement in later development stages.

### Per-Class Performance

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Cardboard | 0.78 | 0.74 | 0.76 |
| Glass | 0.61 | 0.71 | 0.65 |
| Metal | 0.67 | 0.66 | 0.67 |
| Paper | 0.73 | 0.82 | 0.77 |
| Plastic | 0.72 | 0.56 | 0.63 |
| Trash | 0.65 | 0.52 | 0.58 |

The confusion matrix generated during evaluation is stored in:

```text
Project Sprint 1/results/confusion_matrix.png