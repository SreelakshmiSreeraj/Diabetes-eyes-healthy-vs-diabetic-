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
EPOCHS = 10
LEARNING_RATE = 1e-4

BEST_MODEL_PATH = Path(__file__).parent / "best_vit_tiny.pth"


# ==================================================
# LOAD ViT-TINY
# ==================================================

model = timm.create_model(
    "vit_tiny_patch16_224",
    pretrained=True,
    num_classes=NUM_CLASSES
)

model = model.to(DEVICE)


# ==================================================
# LOSS + OPTIMIZER
# ==================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.01
)


# ==================================================
# TRAINING
# ==================================================

best_val_accuracy = 0.0

print("=" * 60)
print("ViT-TINY TRAINING")
print("=" * 60)

print(f"Device       : {DEVICE}")
print(f"Epochs       : {EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")
print("=" * 60)


for epoch in range(EPOCHS):

    # ----------------------------------------------
    # TRAIN
    # ----------------------------------------------

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        # Statistics
        running_loss += loss.item()

        predictions = torch.argmax(outputs, dim=1)

        total += labels.size(0)
        correct += (predictions == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total


    # ----------------------------------------------
    # VALIDATION
    # ----------------------------------------------

    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss += loss.item()

            predictions = torch.argmax(outputs, dim=1)

            val_total += labels.size(0)
            val_correct += (
                predictions == labels
            ).sum().item()

    val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * val_correct / val_total


    # ----------------------------------------------
    # PRINT RESULTS
    # ----------------------------------------------

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


    # ----------------------------------------------
    # SAVE BEST MODEL
    # ----------------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            BEST_MODEL_PATH
        )

        print(
            f"✓ Best model saved "
            f"(Val Accuracy: {val_accuracy:.2f}%)"
        )


# ==================================================
# FINISHED
# ==================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy:.2f}%"
)

print(
    f"Best model saved to:\n"
    f"{BEST_MODEL_PATH}"
)

print("=" * 60)