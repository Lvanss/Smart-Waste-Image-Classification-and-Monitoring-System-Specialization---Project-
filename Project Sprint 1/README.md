# Smart Waste Image Classification and Monitoring System

## 1. Project Overview

The Smart Waste Image Classification and Monitoring System is a computer vision project that classifies waste images into six categories: cardboard, glass, metal, paper, plastic, and trash.

The project uses a Convolutional Neural Network (CNN) trained with the TrashNet dataset. The trained model is exported to ONNX format and deployed through ONNX Runtime, a FastAPI backend, and a Flet desktop graphical user interface.

## 2. Project Objectives

- Develop an image classification model for six waste categories.
- Prepare and organize the dataset for training, validation, and testing.
- Evaluate model performance using classification metrics.
- Export the trained CNN to ONNX format.
- Build an API for image classification.
- Develop a desktop GUI for uploading images and displaying predictions.

## 3. Technologies Used

- Python 3.11
- PyTorch and Torchvision
- NumPy
- Pillow
- scikit-learn
- Matplotlib
- ONNX and ONNX Runtime
- FastAPI and Uvicorn
- Flet
- VS Code

## 4. Dataset

The project uses the TrashNet dataset, containing 2,527 images across six categories.

| Class | Number of Images |
|---|---:|
| Cardboard | 403 |
| Glass | 501 |
| Metal | 410 |
| Paper | 594 |
| Plastic | 482 |
| Trash | 137 |
| **Total** | **2,527** |

The dataset was divided into:

- Training: 70% — 1,765 images
- Validation: 15% — 379 images
- Testing: 15% — 383 images

Images are resized to 128 × 128 pixels and converted to RGB. ImageNet mean and standard deviation normalization are applied during preprocessing.

## 5. Model Architecture

The baseline model is a CNN with three convolutional blocks and a fully connected classification head. It produces six output scores corresponding to the six waste categories.

The model was trained using PyTorch. The best checkpoint is stored at:

`models/checkpoints/baseline_cnn_best.pth`

## 6. Baseline Evaluation Results

The recorded baseline evaluation metrics are:

| Metric | Result |
|---|---:|
| Test Loss | 1.0590 |
| Accuracy | 69.45% |
| Precision | 69.75% |
| Recall | 69.45% |
| F1-Score | 69.23% |

These metrics describe the baseline model's evaluation results. Individual GUI demonstrations should not be interpreted as overall model accuracy.

Detailed evaluation artifacts are stored in the `results/` directory.

## 7. ONNX Export and Verification

The trained PyTorch model is exported to ONNX format using:

`src/model/export_onnx.py`

The exported model is stored at:

`models/onnx/baseline_cnn.onnx`

The export script checks the ONNX model and compares its outputs against the original PyTorch model using ONNX Runtime. The recorded maximum output difference during verification was 0.00000381, and the comparison passed.

## 8. Application Architecture

The application follows this inference pipeline:

1. The user selects a waste image in the Flet desktop GUI.
2. The GUI sends the image to the FastAPI backend.
3. FastAPI passes the image to the ONNX Runtime predictor.
4. The predictor preprocesses the image and executes the ONNX model.
5. The API returns the predicted class, confidence, and inference latency.
6. The GUI displays the classification result.

## 9. Project Structure

```text
Project Sprint 1/
├── Dataset/
├── models/
│   ├── checkpoints/
│   │   └── baseline_cnn_best.pth
│   └── onnx/
│       └── baseline_cnn.onnx
├── notebook/
├── results/
├── src/
│   ├── api/
│   │   └── main.py
│   ├── data/
│   ├── gui/
│   │   └── app.py
│   ├── model/
│   │   ├── baseline_cnn.py
│   │   ├── export_onnx.py
│   │   └── predictor.py
│   └── training/
├── README.md
└── requirements.txt
```

## 10. Installation

Python 3.11 is recommended.

Create and activate a virtual environment from the project directory.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

## 11. Running the Application

Run the following commands from the `Project Sprint 1` directory.

### Start the FastAPI backend

```powershell
python -m uvicorn src.api.main:app --reload
```

The API is available at:

`http://127.0.0.1:8000`

Interactive API documentation:

`http://127.0.0.1:8000/docs`

Keep the API terminal running.

### Start the Flet desktop GUI

Open a second terminal, activate the same virtual environment, and run:

```powershell
python -m src.gui.app
```

Select an image and click **Classify Image** to display the result.

## 12. Testing and Limitations

The application was tested using representative images from all six waste categories. The GUI successfully returned predictions for the demonstrated images, and the FastAPI endpoint returned successful HTTP 200 responses.

One demonstrated plastic image was classified as glass. This illustrates that the model can misclassify images and that individual predictions are not guaranteed to be correct.

The baseline model's recorded test accuracy is 69.45%. Further model improvement and testing with additional unseen images may be needed.

## 13. Future Improvements

- Improve classification performance, especially for visually similar categories.
- Evaluate the model on additional unseen images.
- Improve error handling and input validation.
- Expand the application with waste monitoring and reporting features.
- Investigate deployment to a suitable edge or embedded platform.

## 14. Project Repository

GitHub repository:

https://github.com/Lvanss/Smart-Waste-Image-Classification-and-Monitoring-System-Specialization---Project.git