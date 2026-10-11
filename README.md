# Smart Waste Image Classification and Monitoring System

## Project Overview

The Smart Waste Image Classification and Monitoring System is a computer vision application that classifies a single waste-item image into one of six categories using a Convolutional Neural Network (CNN).

**Waste categories:** Cardboard, Glass, Metal, Paper, Plastic, and Trash.

The project uses the TrashNet dataset, PyTorch for model development, ONNX for model interchange, ONNX Runtime for inference, FastAPI for the prediction API, and Flet for the desktop interface.

## Dataset

The project uses the TrashNet dataset, containing 2,527 images across six categories.

| Class | Images |
|---|---:|
| Cardboard | 403 |
| Glass | 501 |
| Metal | 410 |
| Paper | 594 |
| Plastic | 482 |
| Trash | 137 |
| **Total** | **2,527** |

The dataset split is:

- Training: 1,765 images (70%)
- Validation: 379 images (15%)
- Testing: 383 images (15%)

Raw and processed image files are excluded from Git to avoid storing large dataset files in the repository. Obtain the dataset separately and follow the project's data-preparation scripts and instructions.

## Sprint 1 — Dataset Pipeline and Baseline CNN

Sprint 1 established the machine-learning foundation.

Completed work:
- Dataset inspection, preprocessing, and splitting
- PyTorch dataset-loading pipeline
- Baseline CNN implementation and training
- Test-set evaluation and classification report
- Confusion-matrix and training-history generation
- Git version control setup

### Baseline Model

| Configuration | Value |
|---|---|
| Framework | PyTorch |
| Input | 128 × 128 RGB |
| Output classes | 6 |
| Training device | CPU |
| Epochs | 20 |
| Learning rate | 0.001 |

The baseline CNN contains three convolutional blocks followed by a fully connected classifier.

### Baseline Test Results

| Metric | Result |
|---|---:|
| Test loss | 1.0590 |
| Accuracy | 69.45% |
| Precision | 69.75% |
| Recall | 69.45% |
| F1-score | 69.23% |

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Cardboard | 0.78 | 0.74 | 0.76 |
| Glass | 0.61 | 0.71 | 0.65 |
| Metal | 0.67 | 0.66 | 0.67 |
| Paper | 0.73 | 0.82 | 0.77 |
| Plastic | 0.72 | 0.56 | 0.63 |
| Trash | 0.65 | 0.52 | 0.58 |

The baseline accuracy is below the project's 80% target. These metrics are the baseline evaluation results, not a claim that the target has been achieved.

## Sprint 2 — ONNX Deployment and Application Integration

Sprint 2 integrated the trained model into an image-classification application.

Completed implementation:
- Exported the trained model to ONNX format
- Validated the ONNX model and compared its outputs with PyTorch
- Implemented CPU inference using ONNX Runtime
- Created a FastAPI service for image prediction
- Created a Flet desktop GUI for image selection and prediction display

### Inference Pipeline

1. The user selects a waste-item image in the desktop GUI.
2. The GUI sends the image to the FastAPI `/predict` endpoint.
3. The API validates and processes the uploaded image.
4. The predictor preprocesses the image and runs ONNX Runtime inference.
5. The API returns the predicted category, confidence, and inference latency.
6. The GUI displays the prediction result.

The current interface uses image uploads; webcam-based classification is not implemented.

### Model Export Verification

The ONNX export and verification process completed successfully. The reported maximum output difference between PyTorch and ONNX Runtime was `0.00000381`, and inference used `CPUExecutionProvider`.

### Example Application Tests

Representative sample images from the six categories were tested through the application. These checks demonstrate basic application functionality; they should not be interpreted as the model's overall test accuracy.

## Installation

Use Python 3.11 and run commands from the repository root.

### Sprint 1 — Training and Evaluation Dependencies

```bash
python -m venv .venv
```

Activate the environment, then install:

```bash
pip install -r requirements-sprint1.txt
```

### Sprint 2 — Deployment and GUI Dependencies

With the environment activated, install:

```bash
pip install -r requirements-sprint2.txt
```

The Sprint 2 requirements file includes the Sprint 1 dependencies.

## Running the Application

Run these commands from inside the `Project Sprint` directory, with the virtual environment activated.

### 1. Start the API

```bash
python -m uvicorn src.api.main:app --reload
```

The API is available at `http://127.0.0.1:8000`. Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

### 2. Start the Desktop GUI

Open a second terminal, activate the same environment, navigate to `Project Sprint`, and run:

```bash
python -m src.gui.app
```

Keep the API running while using the GUI.

## Repository Notes

- Dataset images are not included in this repository.
- The ONNX model is used for application inference.
- Checkpoints, generated results, and other artifacts are included only when tracked and available in the repository.
- The project currently uses PyTorch for training and ONNX Runtime for deployment inference.

## Limitations and Future Improvements

- Improve classification performance, especially for categories with lower recall.
- Continue evaluating model performance on held-out and real-world images.
- Add automated tests for API validation, error handling, and inference behavior.
- Consider webcam input and additional deployment options as future enhancements.

## Dependency Files

- `requirements-sprint1.txt` — training, data processing, and evaluation dependencies.
- `requirements-sprint2.txt` — Sprint 1 dependencies plus API, GUI, and ONNX deployment dependencies.
