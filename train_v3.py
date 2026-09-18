from pathlib import Path

import torch
import torch.nn as nn
import timm
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data_loader import train_loader, val_loader


# ==================================================
# SETTINGS
# ==================================================

DEVICE = torch.device("cpu")

NUM_CLASSES = 2
EPOCHS = 15

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.05

PATIENCE = 4

MODEL_PATH = Path(__file__).parent / "best_vit_tiny_v3.pth"


# ==================================================
# LOAD MODEL
# ==================================================

model = timm.create_model(
    "vit_tiny_patch16_224",
    pretrained=True,
    num_classes=NUM_CLASSES
)

model = model.to(DEVICE)


# ==================================================
# FREEZE MODEL
# ==================================================

for param in model.parameters():
    param.requires_grad = False


# Train only final normalization + classification head
for param in model.norm.parameters():
    param.requires_grad = True

for param in model.head.parameters():
    param.requires_grad = True


# ==================================================
# PARAMETER COUNT
# ==================================================

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

total_params = sum(
    p.numel()
    for p in model.parameters()
)


print("=" * 60)
print("ViT-TINY - EXPERIMENT 3")
print("=" * 60)

print(f"Device: {DEVICE}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Total parameters: {total_params:,}")


# ==================================================
# CLASS WEIGHTS
# ==================================================

# Training set:
# Healthy  = 280
# Diabetes = 320

healthy_count = 280
diabetes_count = 320
total_train = healthy_count + diabetes_count

class_weights = torch.tensor(
    [
        total_train / (2 * healthy_count),
        total_train / (2 * diabetes_count)
    ],
    dtype=torch.float32
).to(DEVICE)


# ==================================================
# LOSS
# ==================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=0.1
)


# ==================================================
# OPTIMIZER
# ==================================================

optimizer = AdamW(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ==================================================
# LEARNING RATE SCHEDULER
# ==================================================

scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)


# ==================================================
# TRAINING
# ==================================================

best_val_loss = float("inf")
epochs_without_improvement = 0


for epoch in range(EPOCHS):

    # ------------------------------------------------
    # TRAIN
    # ------------------------------------------------

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        total += labels.size(0)

        correct += (
            predictions == labels
        ).sum().item()

    train_loss = running_loss / len(train_loader)

    train_accuracy = (
        100 * correct / total
    )


    # ------------------------------------------------
    # VALIDATION
    # ------------------------------------------------

    model.eval()

    val_loss_total = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            val_loss_total += loss.item()

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            val_total += labels.size(0)

            val_correct += (
                predictions == labels
            ).sum().item()

    val_loss = (
        val_loss_total / len(val_loader)
    )

    val_accuracy = (
        100 * val_correct / val_total
    )


    # ------------------------------------------------
    # SCHEDULER
    # ------------------------------------------------

    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]["lr"]


    # ------------------------------------------------
    # PRINT RESULTS
    # ------------------------------------------------

    print()
    print(f"Epoch [{epoch + 1}/{EPOCHS}]")

    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.2f}%"
    )

    print(
        f"Val Loss:   {val_loss:.4f} | "
        f"Val Accuracy: {val_accuracy:.2f}%"
    )

    print(
        f"Learning Rate: {current_lr:.6f}"
    )


    # ------------------------------------------------
    # SAVE BEST MODEL
    # ------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        epochs_without_improvement = 0

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print(
            f"✓ Best model saved "
            f"(Val Loss: {val_loss:.4f})"
        )

    else:

        epochs_without_improvement += 1

        print(
            f"No improvement "
            f"({epochs_without_improvement}/{PATIENCE})"
        )


    # ------------------------------------------------
    # EARLY STOPPING
    # ------------------------------------------------

    if epochs_without_improvement >= PATIENCE:

        print()
        print("Early stopping triggered.")

        break


# ==================================================
# FINISHED
# ==================================================

print()
print("=" * 60)
print("EXPERIMENT 3 COMPLETED")
print("=" * 60)

print(
    f"Best validation loss: "
    f"{best_val_loss:.4f}"
)

print(
    "Best model saved to:"
)

print(MODEL_PATH)

print("=" * 60)