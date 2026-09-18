# BioLens AI

## AI-Based Diabetes Risk Classification from Conjunctival Images

BioLens AI is an AI-based research prototype for binary classification of conjunctival eye images into **Healthy** and **Diabetes** classes using a Vision Transformer (ViT-Tiny).

The system also incorporates **ViT-compatible Grad-CAM visualization** to provide visual attribution maps showing image regions that contributed to the model's prediction.

> **Note:** The current implementation focuses specifically on conjunctival images for diabetes classification. The broader BioLens AI concept is intended to support multimodal disease risk prediction from multiple body-image modalities.

---

## Project Overview

Diabetes can cause microvascular changes that may be reflected in the conjunctiva. BioLens AI investigates whether deep learning can identify patterns in conjunctival images associated with diabetes.

The current system performs:

1. Conjunctival image input
2. Image preprocessing
3. ViT-Tiny based classification
4. Binary prediction:
   - Healthy
   - Diabetes
5. Confidence estimation
6. ViT Grad-CAM based visual explanation

---

## System Pipeline

```text
Conjunctival Image
        ↓
Image Preprocessing
        ↓
ViT-Tiny
        ↓
Binary Classification
   ┌───────────────┐
   │               │
Healthy        Diabetes
   │               │
   └───────┬───────┘
           ↓
    Prediction + Confidence
           ↓
   ViT Grad-CAM Visualization
```

---

## Dataset

The project uses the IEEE DataPort dataset:

**Conjunctival and retinal images of healthy subjects and subjects with diabetes**

The dataset contains:

- 108 subjects originally
- 57 subjects with diabetes
- 51 healthy subjects
- 8 conjunctival images per subject
- Additional retinal images for diabetes subjects

For this project, **only conjunctival images are used**. Retinal fundus images and diabetes subgroup information are not used for classification.

The original dataset is **not included in this repository**.

Users should obtain the dataset from the original IEEE DataPort source and place it in the expected directory structure.

---

## Dataset Preprocessing

During dataset verification, two healthy-subject folders contained more than the expected eight conjunctival images.

To maintain a consistent subject/image structure:

- Healthy Subject 19 was excluded from model development because the intended eight images could not be objectively identified.
- The additional `DSC_5407.JPG` image from Healthy Subject 23 was excluded.
- The original dataset files were not deleted or modified.

After preprocessing:

| Category | Subjects | Images |
|---|---:|---:|
| Healthy | 50 | 400 |
| Diabetes | 57 | 456 |
| **Total** | **107** | **856** |

---

## Train / Validation / Test Split

A **subject-level split** was used to prevent images belonging to the same subject from appearing across different splits.

| Split | Subjects | Images |
|---|---:|---:|
| Training | 75 | 600 |
| Validation | 17 | 136 |
| Testing | 15 | 120 |

The test set was kept separate during model development and was used only for final evaluation of each selected experiment.

---

## Model

The classification model is:

**Vision Transformer Tiny (ViT-Tiny)**

Model:

```text
vit_tiny_patch16_224
```

Configuration:

- Input size: 224 × 224
- Number of classes: 2
- Class 0: Healthy
- Class 1: Diabetes
- Total parameters: 5,524,802

The model uses ImageNet-pretrained weights for transfer learning.

---

## Experiments

Three training configurations were evaluated.

### Experiment 1 — Baseline

- Full ViT-Tiny fine-tuning
- Learning rate: `1e-4`
- No training augmentation
- AdamW optimizer

Results:

- Best validation accuracy: **69.12%**
- Test accuracy: **58.33%**
- Test precision: **0.6842**
- Diabetes recall: **0.4062**
- F1-score: **0.5098**

The model showed substantial overfitting, with training accuracy reaching 100% while validation performance did not improve accordingly.

---

### Experiment 2 — Regularized Partial Fine-Tuning

Changes included:

- Training augmentation
- Partial ViT fine-tuning
- Lower learning rate: `2e-5`
- Class-weighted loss
- Weight decay: `0.05`
- Early stopping
- ReduceLROnPlateau learning-rate scheduling

Results:

- Best validation accuracy: **75.00%**
- Test accuracy: **66.67%**
- Test precision: **0.7400**
- Diabetes recall: **0.5781**
- F1-score: **0.6491**

Experiment 2 was selected as the final model configuration based on the validation and held-out test results.

---

### Experiment 3 — Frozen Feature Extractor

Changes included:

- Frozen ViT transformer blocks
- Only final normalization and classification head trained
- Label smoothing
- Class-weighted loss

Trainable parameters:

```text
770
```

Results:

- Best validation accuracy: **73.53%**
- Test accuracy: **56.67%**
- Test precision: **0.5968**
- Diabetes recall: **0.5781**
- F1-score: **0.5873**

Experiment 3 did not outperform Experiment 2 on the held-out test set.

---

## Final Model

The selected model is the Experiment 2 checkpoint:

```text
best_vit_tiny_v2.pth
```

Final held-out test performance:

| Metric | Result |
|---|---:|
| Accuracy | **66.67%** |
| Precision | **0.7400** |
| Diabetes Recall | **0.5781** |
| F1-score | **0.6491** |
| Test Loss | **0.6900** |

### Confusion Matrix

```text
                 Predicted
              Healthy  Diabetes

Actual Healthy    43       13
Actual Diabetes   27       37
```

---

## Explainability

The project uses a ViT-compatible Grad-CAM approach to visualize regions that contributed to the model's prediction.

Representative cases include:

- Correct Healthy prediction
- Correct Diabetes prediction
- Healthy predicted as Diabetes
- Diabetes predicted as Healthy

Generated explanations contain:

```text
Original Image
       ↓
ViT Grad-CAM
       ↓
Explanation Overlay
```

The heatmaps represent **model attribution** and should not be interpreted as proof of medical causation or as clinically validated diagnostic evidence.

---

## Application

A Flask-based web application is included.

The application allows the user to:

1. Upload a conjunctival image
2. Run the trained ViT-Tiny model
3. View the predicted class
4. View prediction confidence
5. View the Grad-CAM heatmap
6. View the explanation overlay

Run the application using:

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

---

## Project Structure

```text
Diabeteseyes/
│
├── dataset/                  # Local dataset - not included in GitHub
├── uploads/                  # Generated uploads - not included
├── explainability_results/   # Generated explanations - not included
│
├── app.py                    # Flask application
├── model.py                  # ViT-Tiny model setup
├── data_loader.py            # Dataset and DataLoader
│
├── prepare_dataset.py        # Dataset manifest generation
├── split_dataset.py          # Subject-level dataset splitting
├── check_dataset.py          # Dataset verification
│
├── train.py                  # Experiment 1
├── train_v2.py               # Experiment 2
├── train_v3.py               # Experiment 3
│
├── evaluate.py               # Model evaluation
├── explain.py                # ViT Grad-CAM explainability
│
├── dataset_manifest.csv      # Dataset manifest
├── dataset_split.csv         # Fixed train/validation/test split
│
├── best_vit_tiny.pth         # Experiment 1 checkpoint
├── best_vit_tiny_v2.pth      # Final selected checkpoint
├── best_vit_tiny_v3.pth      # Experiment 3 checkpoint
│
└── .gitignore
```

---

## Installation

Create a Python environment and install the required packages:

```bash
pip install torch torchvision timm pillow flask matplotlib numpy
```

The project was developed and tested using Python 3.12 and CPU-based PyTorch.

---

## Running the Project

### Train

Experiment 1:

```bash
python train.py
```

Experiment 2:

```bash
python train_v2.py
```

Experiment 3:

```bash
python train_v3.py
```

### Evaluate

```bash
python evaluate.py
```

### Explainability

```bash
python explain.py
```

### Web Application

```bash
python app.py
```

---

## Important Notes

- The dataset is not included in this repository.
- The project uses a fixed subject-level train/validation/test split.
- Fundus images are not used by the current classification system.
- The current model performs binary Healthy vs Diabetes classification.
- The system is a research prototype and has not been clinically validated.
- Predictions should not be used as a substitute for professional medical diagnosis.

---

## Future Work

Future development may include:

- Additional disease categories
- Additional body-image modalities
- Multimodal feature fusion
- Larger and more diverse datasets
- Multi-site data collection
- Improved image quality and lighting robustness
- Further explainability methods
- External validation on independent datasets
- Development of a more comprehensive multimodal BioLens AI framework

---

## Project

**BioLens AI**

AI-based disease risk prediction using medical image analysis and explainable artificial intelligence.

**Current implementation:** Diabetes classification from conjunctival images using ViT-Tiny.
