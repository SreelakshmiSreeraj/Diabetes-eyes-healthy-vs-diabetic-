import torch
import timm


# Number of classes
NUM_CLASSES = 2

# Create ViT-Tiny model
model = timm.create_model(
    "vit_tiny_patch16_224",
    pretrained=True,
    num_classes=NUM_CLASSES
)

print("=" * 50)
print("ViT-Tiny MODEL")
print("=" * 50)

print(model)

print("\nModel successfully loaded!")
print(f"Number of classes: {NUM_CLASSES}")
print("Class 0: Healthy")
print("Class 1: Diabetes")

# Check device
device = torch.device("cpu")
model = model.to(device)

print(f"\nDevice: {device}")