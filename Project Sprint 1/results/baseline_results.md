# Smart Waste Image Classification and Monitoring System
## Sprint 1 - Baseline CNN Results

### Dataset

Dataset: TrashNet

Total images: 2,527

Classes:
- cardboard
- glass
- metal
- paper
- plastic
- trash

Dataset split:
- Training: 1,765 images (70%)
- Validation: 379 images (15%)
- Testing: 383 images (15%)

### Baseline Model

Model: BaselineCNN

Framework: PyTorch

Input size: 128 × 128 RGB

Number of output classes: 6

Training device: CPU

Number of epochs: 20

Learning rate: 0.001

### Training Result

Best validation accuracy: 73.09%

Final training accuracy: 82.49%

### Test Result

Test loss: 1.0590

Test accuracy: 69.45%

Precision: 69.75%

Recall: 69.45%

F1-score: 69.23%

### Per-Class Performance

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Cardboard | 0.78 | 0.74 | 0.76 | 61 |
| Glass | 0.61 | 0.71 | 0.65 | 76 |
| Metal | 0.67 | 0.66 | 0.67 | 62 |
| Paper | 0.73 | 0.82 | 0.77 | 90 |
| Plastic | 0.72 | 0.56 | 0.63 | 73 |
| Trash | 0.65 | 0.52 | 0.58 | 21 |

### Confusion Matrix

The confusion matrix is saved as:

`results/confusion_matrix.png`

### Sprint 1 Interpretation

The baseline CNN successfully completed training and testing on the six-class TrashNet dataset. The model achieved 69.45% test accuracy and provides a functional baseline for subsequent model improvement.

The results indicate that some classes are more difficult to distinguish than others. Plastic and trash have comparatively lower recall, while paper achieved the highest recall among the six classes. These results will be used to guide model refinement in later development.

### Sprint 1 Status

- [x] Dataset inspected
- [x] Dataset cleaned/validated
- [x] Dataset split into training, validation, and testing sets
- [x] PyTorch DataLoader implemented
- [x] Baseline CNN implemented
- [x] Model input/output dimensions verified
- [x] Baseline model trained
- [x] Test evaluation completed
- [x] Precision, recall, and F1-score calculated
- [x] Confusion matrix generated
- [x] Model checkpoint saved
- [x] Training history saved
- [x] Evaluation results saved