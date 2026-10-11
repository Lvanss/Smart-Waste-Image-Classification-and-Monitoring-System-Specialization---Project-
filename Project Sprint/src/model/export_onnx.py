from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from src.model.baseline_cnn import BaselineCNN


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = (
    PROJECT_ROOT / "models" / "checkpoints" / "baseline_cnn_best.pth"
)
ONNX_DIR = PROJECT_ROOT / "models" / "onnx"
ONNX_PATH = ONNX_DIR / "baseline_cnn.onnx"

# Model configuration
IMAGE_SIZE = 128
NUM_CLASSES = 6


def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {CHECKPOINT_PATH}"
        )

    ONNX_DIR.mkdir(parents=True, exist_ok=True)

    # Load the trained PyTorch model
    model = BaselineCNN(num_classes=NUM_CLASSES)
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Create a sample input: batch of one 128x128 RGB image
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)

    # Export the model to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        str(ONNX_PATH),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=17,
        dynamo=False,
    )

    print(f"\nONNX model exported to:\n{ONNX_PATH}")

    # Validate the exported ONNX file
    onnx_model = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(onnx_model)
    print("ONNX model validation: PASSED")

    # Compare PyTorch and ONNX Runtime outputs
    session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=["CPUExecutionProvider"],
    )

    sample = np.random.randn(
        1, 3, IMAGE_SIZE, IMAGE_SIZE
    ).astype(np.float32)

    with torch.no_grad():
        pytorch_output = model(torch.from_numpy(sample)).numpy()

    onnx_output = session.run(
        ["logits"],
        {"input": sample},
    )[0]

    max_difference = np.max(
        np.abs(pytorch_output - onnx_output)
    )

    print(f"ONNX Runtime provider: {session.get_providers()[0]}")
    print(f"Maximum output difference: {max_difference:.8f}")

    if np.allclose(
        pytorch_output,
        onnx_output,
        rtol=1e-3,
        atol=1e-4,
    ):
        print("PyTorch vs ONNX Runtime comparison: PASSED")
    else:
        raise RuntimeError(
            "PyTorch and ONNX Runtime outputs do not match closely enough."
        )

    print("\nONNX export and verification completed successfully.")


if __name__ == "__main__":
    main()