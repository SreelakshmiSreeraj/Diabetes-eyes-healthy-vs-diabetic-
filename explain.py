from pathlib import Path
import csv

import numpy as np
import torch
import torch.nn.functional as F
import timm

from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt


# ==================================================
# SETTINGS
# ==================================================

DEVICE = torch.device("cpu")

NUM_CLASSES = 2

PROJECT_PATH = Path(__file__).parent

MODEL_PATH = PROJECT_PATH / "best_vit_tiny_v2.pth"
SPLIT_FILE = PROJECT_PATH / "dataset_split.csv"

RESULTS_DIR = PROJECT_PATH / "explainability_results"
RESULTS_DIR.mkdir(exist_ok=True)


CLASS_NAMES = [
    "Healthy",
    "Diabetes"
]


# ==================================================
# IMAGE TRANSFORM
# ==================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==================================================
# LOAD MODEL
# ==================================================

print("=" * 60)
print("LOADING FINAL ViT-TINY MODEL")
print("=" * 60)

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

print("Model loaded successfully.")


# ==================================================
# LOAD TEST IMAGES
# ==================================================

test_samples = []

with open(
    SPLIT_FILE,
    "r",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        if row["split"] == "test":

            test_samples.append(row)


print(f"Test images: {len(test_samples)}")


# ==================================================
# FIND REPRESENTATIVE IMAGES
# ==================================================

correct_healthy = None
correct_diabetes = None

wrong_healthy = None
wrong_diabetes = None


print()
print("Scanning test set...")


for row in test_samples:

    image_path = Path(row["image_path"])

    true_label_name = row["label"]

    true_label = (
        0 if true_label_name == "Healthy"
        else 1
    )

    image = Image.open(
        image_path
    ).convert("RGB")

    image_tensor = transform(
        image
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        prediction = torch.argmax(
            outputs,
            dim=1
        ).item()

        confidence = probabilities[
            0,
            prediction
        ].item()


    # Correct Healthy
    if (
        true_label == 0
        and prediction == 0
        and correct_healthy is None
    ):

        correct_healthy = {
            "row": row,
            "confidence": confidence
        }


    # Correct Diabetes
    elif (
        true_label == 1
        and prediction == 1
        and correct_diabetes is None
    ):

        correct_diabetes = {
            "row": row,
            "confidence": confidence
        }


    # Healthy predicted as Diabetes
    elif (
        true_label == 0
        and prediction == 1
        and wrong_healthy is None
    ):

        wrong_healthy = {
            "row": row,
            "confidence": confidence
        }


    # Diabetes predicted as Healthy
    elif (
        true_label == 1
        and prediction == 0
        and wrong_diabetes is None
    ):

        wrong_diabetes = {
            "row": row,
            "confidence": confidence
        }


# ==================================================
# DISPLAY SELECTED IMAGES
# ==================================================

print()
print("=" * 60)
print("REPRESENTATIVE TEST IMAGES")
print("=" * 60)


categories = [
    ("correct_healthy", "Correct Healthy", correct_healthy),
    ("correct_diabetes", "Correct Diabetes", correct_diabetes),
    ("healthy_as_diabetes", "Healthy → Diabetes", wrong_healthy),
    ("diabetes_as_healthy", "Diabetes → Healthy", wrong_diabetes)
]


for filename, description, result in categories:

    if result is None:

        print(f"{description}: NOT FOUND")

    else:

        print(
            f"{description}: "
            f"{result['row']['image_path']} "
            f"({result['confidence'] * 100:.2f}%)"
        )


# ==================================================
# GRAD-CAM FUNCTION
# ==================================================

def generate_gradcam(
    image_path,
    true_label_name,
    output_path
):

    global activations
    global gradients

    activations = None
    gradients = None


    # ----------------------------------------------
    # Load image
    # ----------------------------------------------

    original_image = Image.open(
        image_path
    ).convert("RGB")


    input_tensor = transform(
        original_image
    ).unsqueeze(0).to(DEVICE)


    # ----------------------------------------------
    # Hooks
    # ----------------------------------------------

    target_layer = model.blocks[-1].norm1


    def forward_hook(module, input, output):

        global activations

        activations = output


    def backward_hook(
        module,
        grad_input,
        grad_output
    ):

        global gradients

        gradients = grad_output[0]


    forward_handle = (
        target_layer.register_forward_hook(
            forward_hook
        )
    )

    backward_handle = (
        target_layer.register_full_backward_hook(
            backward_hook
        )
    )


    # ----------------------------------------------
    # Forward pass
    # ----------------------------------------------

    model.zero_grad()

    outputs = model(input_tensor)

    prediction = torch.argmax(
        outputs,
        dim=1
    ).item()


    probabilities = torch.softmax(
        outputs,
        dim=1
    )

    confidence = probabilities[
        0,
        prediction
    ].item()


    # ----------------------------------------------
    # Backward pass
    # ----------------------------------------------

    target_score = outputs[
        0,
        prediction
    ]

    target_score.backward()


    # ----------------------------------------------
    # Extract activations/gradients
    # ----------------------------------------------

    acts = activations[0]
    grads = gradients[0]


    # Remove CLS token

    acts = acts[1:]
    grads = grads[1:]


    # Average gradients

    weights = grads.mean(
        dim=0
    )


    # Weighted activation

    cam = torch.sum(
        acts * weights,
        dim=1
    )


    # ReLU

    cam = F.relu(cam)


    # ViT-Tiny:
    # 14 × 14 = 196 patches

    cam = cam.reshape(
        14,
        14
    )


    # Normalize

    cam = cam - cam.min()

    if cam.max() > 0:

        cam = cam / cam.max()


    cam = cam.detach().cpu().numpy()


    # ----------------------------------------------
    # Resize heatmap
    # ----------------------------------------------

    heatmap = Image.fromarray(
        (cam * 255).astype(np.uint8)
    )

    heatmap = heatmap.resize(
        original_image.size,
        Image.Resampling.BILINEAR
    )


    heatmap_array = (
        np.array(heatmap) / 255.0
    )

    original_array = (
        np.array(original_image) / 255.0
    )


    # ----------------------------------------------
    # Create figure
    # ----------------------------------------------

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )


    # Original

    axes[0].imshow(
        original_image
    )

    axes[0].set_title(
        f"Original\nTrue: {true_label_name}"
    )

    axes[0].axis("off")


    # Heatmap

    axes[1].imshow(
        heatmap_array,
        cmap="jet"
    )

    axes[1].set_title(
        "ViT Grad-CAM"
    )

    axes[1].axis("off")


    # Overlay

    axes[2].imshow(
        original_array
    )

    axes[2].imshow(
        heatmap_array,
        cmap="jet",
        alpha=0.45
    )

    axes[2].set_title(
        f"Prediction: {CLASS_NAMES[prediction]}\n"
        f"Confidence: {confidence * 100:.1f}%"
    )

    axes[2].axis("off")


    plt.tight_layout()


    # ----------------------------------------------
    # Save
    # ----------------------------------------------

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


    # ----------------------------------------------
    # Remove hooks
    # ----------------------------------------------

    forward_handle.remove()
    backward_handle.remove()


    return prediction, confidence


# ==================================================
# GENERATE EXPLANATIONS
# ==================================================

print()
print("=" * 60)
print("GENERATING ViT GRAD-CAM RESULTS")
print("=" * 60)


generated = 0


for filename, description, result in categories:

    if result is None:

        continue


    row = result["row"]

    image_path = Path(
        row["image_path"]
    )

    true_label = row["label"]


    output_path = (
        RESULTS_DIR /
        f"{filename}.png"
    )


    prediction, confidence = generate_gradcam(
        image_path,
        true_label,
        output_path
    )


    print()
    print(f"✓ {description}")
    print(f"  True: {true_label}")
    print(
        f"  Predicted: "
        f"{CLASS_NAMES[prediction]}"
    )
    print(
        f"  Confidence: "
        f"{confidence * 100:.2f}%"
    )
    print(
        f"  Saved: {output_path}"
    )


    generated += 1


# ==================================================
# FINISHED
# ==================================================

print()
print("=" * 60)
print("EXPLAINABILITY COMPLETED")
print("=" * 60)

print(
    f"Generated explanations: "
    f"{generated}"
)

print(
    f"Results folder:\n"
    f"{RESULTS_DIR}"
)

print("=" * 60)