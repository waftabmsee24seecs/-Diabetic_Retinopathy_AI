
Simulation and embedded code
The eye_diseases_detection.ipynb file inlcudes fine-tuning efficientnet-B0 model for eye disease detection.
Files Overview
1. Dataset Creation – Retinal vs Non-Retinal

Dataset_Creation_retinal_nonretinal.py
Builds a balanced binary dataset (retinal / non-retinal)
Retinal sources: EyePACS, APTOS, Messidor (via KaggleHub)
Non-retinal sources: CIFAR-10, CIFAR-100, COCO-2017
Produces a fully shuffled dataset:
Train & Validation splits
Outputs a zipped dataset for fast reuse and deployment

Binary_Retina_Nonretina_Classification.py
EfficientNet-B3 based binary classifier
Mixed precision training for speed and memory efficiency
Two-stage training:
    Stage 1: frozen backbone
    Stage 2: full fine-tuning
    Uses tf.data pipelines for optimized I/O
Outputs:
    retina_model_best.h5 (deployment-ready)
    retina_model_final.h5

Diabetic_Retinopathy_Severity_Classification.py
EfficientNet-B4 based 5-class DR severity classifier
Memory-stable multi-stage fine-tuning strategy
Handles class imbalance using class weights
Advanced evaluation metrics:
AUC
Quadratic Weighted Kappa (QWK)
Saves:
  Model weights
  Model architecture (JSON) for reproducibility

app.py
Inference API / application entry point
Loads trained models for:
    Retina detection
    DR severity classification
Designed for integration with deployment platforms (e.g., Flask / Streamlit / Jetson Nano)

