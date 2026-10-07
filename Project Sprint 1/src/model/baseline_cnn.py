import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    """
    Baseline CNN for six-class TrashNet image classification.

    Input:
        3 x 128 x 128 RGB image

    Output:
        6 class logits
    """

    def __init__(self, num_classes=6):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Block 2
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Block 3
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(
                in_features=128 * 16 * 16,
                out_features=128
            ),

            nn.ReLU(),

            nn.Dropout(p=0.5),

            nn.Linear(
                in_features=128,
                out_features=num_classes
            )
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------
# Model verification
# ---------------------------------------------------------

if __name__ == "__main__":

    model = BaselineCNN()

    # Simulate one batch from the dataset loader
    sample_input = torch.randn(32, 3, 128, 128)

    output = model(sample_input)

    print("=" * 60)
    print("BASELINE CNN MODEL TEST")
    print("=" * 60)

    print("\nModel architecture:")
    print(model)

    print("\nInput shape:")
    print(sample_input.shape)

    print("\nOutput shape:")
    print(output.shape)

    print("\nNumber of parameters:")
    print(sum(p.numel() for p in model.parameters()))

    print("\nExpected output:")
    print("32 images x 6 classes")

    if output.shape == (32, 6):
        print("\nSUCCESS: Model input/output dimensions are correct.")
    else:
        print("\nWARNING: Unexpected model output dimensions.")

    print("\n" + "=" * 60)
    print("BASELINE CNN TEST COMPLETE")
    print("=" * 60)