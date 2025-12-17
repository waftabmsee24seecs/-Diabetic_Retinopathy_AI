
Figures, tables, screenshots, logs, and evaluation outputs.
EfficientNet-B4 with Staged Fine-Tuning & Quadratic Weighted Kappa Optimization.
Training Strategy

Training was performed using staged fine-tuning, progressively unfreezing the network while reducing the learning rate to prevent catastrophic forgetting and overfitting.

Stage 1 — Head Training
Backbone frozen
Only classification head trained
Learning Rate: 1e-3
Epochs: 4
Stage 2 — Partial Fine-Tuning
Upper layers of EfficientNet unfrozen
Learning Rate: 5e-5
Epochs: 4
Stage 3 — Full Fine-Tuning
Entire network unfrozen
Learning Rate: 1e-5 → dynamically reduced using ReduceLROnPlateau
Epochs: 7


