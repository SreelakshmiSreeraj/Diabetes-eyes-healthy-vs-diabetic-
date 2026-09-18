from pathlib import Path

import torch
import torch.nn as nn
import timm

from data_loader import test_loader


# ==================================================
# SETTINGS
# ==================================================

DEVICE = torch.device("cpu")

NUM_CLASSES = 2

# Experiment 2 best model
MODEL_PATH = Path(__file__).parent / "best_vit_tiny_v3.pth"

# ==================================================
# LOAD MODEL
# ==================================================

model = timm.create_model(
    "vit_tiny_patch16_224",
    pretrained=False,
    num_classes=NUM_CLASSES
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model = model.to(DEVICE)
model.eval()


# ==================================================
# EVALUATION
# ==================================================

criterion = nn.CrossEntropyLoss()

total_loss = 0.0
correct = 0
total = 0

# Confusion matrix:
#
#             Predicted
#             Healthy Diabetes
# Actual
# Healthy       TN      FP
# Diabetes      FN      TP

true_negative = 0
false_positive = 0
false_negative = 0
true_positive = 0


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        loss = criterion(outputs, labels)

        total_loss += loss.item()

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        total += labels.size(0)

        correct += (
            predictions == labels
        ).sum().item()

        # Update confusion matrix
        for actual, predicted in zip(
            labels.tolist(),
            predictions.tolist()
        ):

            if actual == 0 and predicted == 0:
                true_negative += 1

            elif actual == 0 and predicted == 1:
                false_positive += 1

            elif actual == 1 and predicted == 0:
                false_negative += 1

            elif actual == 1 and predicted == 1:
                true_positive += 1


# ==================================================
# METRICS
# ==================================================

test_loss = total_loss / len(test_loader)

accuracy = (
    100 * correct / total
)

precision = (
    true_positive /
    (true_positive + false_positive)
    if (true_positive + false_positive) > 0
    else 0
)

recall = (
    true_positive /
    (true_positive + false_negative)
    if (true_positive + false_negative) > 0
    else 0
)

f1 = (
    2 * precision * recall /
    (precision + recall)
    if (precision + recall) > 0
    else 0
)


# ==================================================
# RESULTS
# ==================================================

print("=" * 60)
print("ViT-TINY TEST SET EVALUATION")
print("=" * 60)

print(f"Test images : {total}")
print(f"Test loss   : {test_loss:.4f}")
print(f"Accuracy    : {accuracy:.2f}%")

print("\nCONFUSION MATRIX")
print("-" * 40)

print("                 Predicted")
print("              Healthy  Diabetes")
print(
    f"Actual Healthy   {true_negative:3d}       {false_positive:3d}"
)
print(
    f"Actual Diabetes  {false_negative:3d}       {true_positive:3d}"
)

print("\nCLASSIFICATION METRICS")
print("-" * 40)

print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")

print("\n" + "=" * 60)
print("EVALUATION COMPLETED")
print("=" * 60)