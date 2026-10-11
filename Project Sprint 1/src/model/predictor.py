
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "onnx" / "baseline_cnn.onnx"

CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
]

IMAGE_SIZE = 128


# Must match the preprocessing used during evaluation
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


class WastePredictor:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"ONNX model not found: {MODEL_PATH}"
            )

        # Load the ONNX model using CPU inference
        self.session = ort.InferenceSession(
            str(MODEL_PATH),
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, image: Image.Image) -> dict:
        """Classify a PIL image using ONNX Runtime."""

        start_time = time.perf_counter()

        image = image.convert("RGB")
        input_tensor = IMAGE_TRANSFORM(image).unsqueeze(0)

        # Convert the PyTorch tensor to a NumPy array for ONNX Runtime
        input_array = input_tensor.numpy().astype(np.float32)

        # Run inference
        logits = self.session.run(
            [self.output_name],
            {self.input_name: input_array},
        )[0]

        # Convert model outputs to class probabilities
        logits = logits[0]
        shifted_logits = logits - np.max(logits)
        exp_logits = np.exp(shifted_logits)
        probabilities = exp_logits / np.sum(exp_logits)

        predicted_index = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_index])

        latency_ms = (time.perf_counter() - start_time) * 1000

        return {
            "predicted_class": CLASS_NAMES[predicted_index],
            "confidence": round(confidence, 6),
            "confidence_percent": round(confidence * 100, 2),
            "inference_latency_ms": round(latency_ms, 2),
            "model": "BaselineCNN-ONNX",
        }


if __name__ == "__main__":
    print("Loading Smart Waste Image Classifier...")

    predictor = WastePredictor()

    print(f"ONNX model: {MODEL_PATH}")
    print(f"Execution providers: {predictor.session.get_providers()}")
    print(f"Classes: {CLASS_NAMES}")
    print("Model loaded successfully.")

    image_path = input(
        "\nEnter the full path to a waste image: "
    ).strip().strip('"')

    if not Path(image_path).is_file():
        raise FileNotFoundError(
            f"Image file not found: {image_path}"
        )

    with Image.open(image_path) as image:
        result = predictor.predict(image)

    print("\n" + "=" * 45)
    print("WASTE CLASSIFICATION RESULT")
    print("=" * 45)
    print(f"Predicted class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence_percent']}%")
    print(f"Inference latency: {result['inference_latency_ms']} ms")
    print(f"Model: {result['model']}")
    print("=" * 45)