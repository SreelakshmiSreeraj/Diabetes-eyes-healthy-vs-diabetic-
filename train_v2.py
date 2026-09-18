from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import timm

from data_loader import train_loader, val_loader


# ==================================================
# SETTINGS
# ==================================================

DEVICE = torch.device("cpu")

NUM_CLASSES = 2
EPOCHS = 15
LEARNING_RATE = 2e-5
PATIENCE = 4

BEST_MODEL_PATH = (
    Path(__file__).parent / "best_vit_tiny_v2.pth"
)


# ==================================================
# LOAD PRETRAINED ViT-TINY
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

# Freeze all parameters first
for parameter in model.parameters():
    parameter.requires_grad = False


# Unfreeze the last two transformer blocks
for block in model.blocks[-2:]:
    for parameter in block.parameters():
        parameter.requires_grad = True


# Unfreeze final normalization layer
for parameter in model.norm.parameters():
    parameter.requires_grad = True


# Unfreeze classification head
for parameter in model.head.parameters():
    parameter.requires_grad = True


# ==================================================
# COUNT TRAINABLE PARAMETERS
# ==================================================

trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print("=" * 60)
print("ViT-TINY - EXPERIMENT 2")
print("=" * 60)

print(f"Device: {DEVICE}")

print(
    f"Trainable parameters: "
    f"{trainable_parameters:,}"
)

print(
    f"Total parameters: "
    f"{total_parameters:,}"
)


# ==================================================
# LOSS FUNCTION
# ==================================================

# Dataset has slightly more Diabetes images than Healthy.
# We use class weights calculated from the training set.

healthy_count = 280
diabetes_count = 320

total_train = healthy_count + diabetes_count

healthy_weight = total_train / (2 * healthy_count)
diabetes_weight = total_train / (2 * diabetes_count)

class_weights = torch.tensor(
    [healthy_weight, diabetes_weight],
    dtype=torch.float32
).to(DEVICE)

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ==================================================
# OPTIMIZER
# ==================================================

optimizer = optim.AdamW(
    filter(
        lambda parameter: parameter.requires_grad,
        model.parameters()
    ),
    lr=LEARNING_RATE,
    weight_decay=0.05
)


# ==================================================
# LEARNING RATE SCHEDULER
# ==================================================

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)


# ==================================================
# TRAINING VARIABLES
# ==================================================

best_val_loss = float("inf")
epochs_without_improvement = 0


# ==================================================
# TRAINING LOOP
# ==================================================

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

        loss = criterion(
            outputs,
            labels
        )

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

    train_loss = (
        running_loss /
        len(train_loader)
    )

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
        val_loss_total /
        len(val_loader)
    )

    val_accuracy = (
        100 * val_correct / val_total
    )


    # ------------------------------------------------
    # SCHEDULER
    # ------------------------------------------------

    scheduler.step(val_loss)


    # ------------------------------------------------
    # PRINT RESULTS
    # ------------------------------------------------

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"\nEpoch [{epoch + 1}/{EPOCHS}]"
    )

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
            BEST_MODEL_PATH
        )

        print(
            "✓ Best model saved "
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

        print(
            "\nEarly stopping triggered."
        )

        break


# ==================================================
# FINISHED
# ==================================================

print("\n" + "=" * 60)
print("EXPERIMENT 2 COMPLETED")
print("=" * 60)

print(
    f"Best validation loss: "
    f"{best_val_loss:.4f}"
)

print(
    f"Best model saved to:\n"
    f"{BEST_MODEL_PATH}"
)

print("=" * 60)`