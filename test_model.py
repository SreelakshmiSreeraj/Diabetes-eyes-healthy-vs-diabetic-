import torch
import timm

from data_loader import train_loader


# ==========================================
# DEVICE
# ==========================================

device = torch.device("cpu")


# ==========================================
# LOAD ViT-TINY
# ==========================================

model = timm.create_model(
    "vit_tiny_patch16_224",
    pretrained=True,
    num_classes=2
)

model = model.to(device)
model.eval()


# ==========================================
# GET ONE BATCH
# ==========================================

images, labels = next(iter(train_loader))

images = images.to(device)
labels = labels.to(device)


# ==========================================
# FORWARD PASS
# ==========================================

with torch.no_grad():

    outputs = model(images)


# ==========================================
# RESULTS
# ==========================================

print("=" * 55)
print("ViT-TINY FORWARD PASS TEST")
print("=" * 55)

print(f"Input shape  : {images.shape}")
print(f"Output shape : {outputs.shape}")
print(f"Labels shape : {labels.shape}")

print("\nModel output:")
print(outputs)

predictions = torch.argmax(outputs, dim=1)

print("\nPredictions:")
print(predictions.tolist())

print("\nActual labels:")
print(labels.tolist())

print("\n" + "=" * 55)
print("FORWARD PASS SUCCESSFUL")
print("=" * 55)