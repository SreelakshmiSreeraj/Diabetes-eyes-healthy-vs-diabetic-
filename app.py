from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import timm

from PIL import Image
from torchvision import transforms

from flask import (
    Flask,
    render_template_string,
    request
)


# ==================================================
# SETTINGS
# ==================================================

DEVICE = torch.device("cpu")

PROJECT_PATH = Path(__file__).parent

MODEL_PATH = (
    PROJECT_PATH /
    "best_vit_tiny_v2.pth"
)


UPLOAD_FOLDER = (
    PROJECT_PATH /
    "uploads"
)

UPLOAD_FOLDER.mkdir(
    exist_ok=True
)


# ==================================================
# FLASK APP
# ==================================================

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = str(
    UPLOAD_FOLDER
)


# ==================================================
# CLASS NAMES
# ==================================================

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
# LOAD FINAL ViT-TINY MODEL
# ==================================================

print("=" * 60)
print("LOADING BioLens AI MODEL")
print("=" * 60)

model = timm.create_model(
    "vit_tiny_patch16_224",
    pretrained=False,
    num_classes=2
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model = model.to(DEVICE)
model.eval()

print("Final ViT-Tiny model loaded.")
print("=" * 60)


# ==================================================
# GRAD-CAM FUNCTION
# ==================================================

def generate_gradcam(
    image,
    input_tensor,
    prediction
):

    activations = None
    gradients = None


    # ----------------------------------------------
    # Target layer
    # ----------------------------------------------

    target_layer = model.blocks[-1].norm1


    def forward_hook(
        module,
        input,
        output
    ):

        nonlocal activations

        activations = output


    def backward_hook(
        module,
        grad_input,
        grad_output
    ):

        nonlocal gradients

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

    outputs = model(
        input_tensor
    )


    # ----------------------------------------------
    # Backward pass
    # ----------------------------------------------

    target_score = outputs[
        0,
        prediction
    ]

    target_score.backward()


    # ----------------------------------------------
    # Get activations
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


    # Weighted activations

    cam = torch.sum(
        acts * weights,
        dim=1
    )


    # ReLU

    cam = F.relu(cam)


    # ViT-Tiny:
    # 196 patches = 14 × 14

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
        image.size,
        Image.Resampling.BILINEAR
    )


    # ----------------------------------------------
    # Create overlay
    # ----------------------------------------------

    original_array = (
        np.array(image)
    )

    heatmap_array = (
        np.array(heatmap)
    )


    # Convert heatmap to RGB using
    # matplotlib-style jet mapping manually

    import matplotlib.cm as cm

    colored_heatmap = (
        cm.jet(
            heatmap_array / 255.0
        )[:, :, :3] * 255
    ).astype(np.uint8)


    colored_heatmap = Image.fromarray(
        colored_heatmap
    )


    overlay = Image.blend(
        image.convert("RGB"),
        colored_heatmap,
        alpha=0.45
    )


    # ----------------------------------------------
    # Save images
    # ----------------------------------------------

    original_path = (
        UPLOAD_FOLDER /
        "original.jpg"
    )

    heatmap_path = (
        UPLOAD_FOLDER /
        "heatmap.jpg"
    )

    overlay_path = (
        UPLOAD_FOLDER /
        "overlay.jpg"
    )


    image.save(
        original_path
    )

    heatmap.save(
        heatmap_path
    )

    overlay.save(
        overlay_path
    )

    forward_handle.remove()
    backward_handle.remove()


    return (
        original_path,
        heatmap_path,
        overlay_path
    )


# ==================================================
# HTML
# ==================================================

HTML = """
<!DOCTYPE html>

<html>

<head>

<title>BioLens AI</title>

<style>

body {

    font-family: Arial, sans-serif;

    background:
        linear-gradient(
            135deg,
            #eef4ff,
            #f8fbff
        );

    margin: 0;

    padding: 0;

    text-align: center;

}


.container {

    max-width: 1000px;

    margin: 40px auto;

    background: white;

    padding: 35px;

    border-radius: 20px;

    box-shadow:
        0 8px 30px
        rgba(0,0,0,0.08);

}


h1 {

    font-size: 38px;

    margin-bottom: 5px;

}


.subtitle {

    color: #666;

    margin-bottom: 30px;

}


input[type=file] {

    margin: 20px;

}


button {

    background: #2563eb;

    color: white;

    border: none;

    padding: 12px 25px;

    border-radius: 10px;

    font-size: 16px;

    cursor: pointer;

}


button:hover {

    background: #1d4ed8;

}


.result {

    margin-top: 30px;

    padding: 20px;

    border-radius: 15px;

    background: #f5f7fb;

}


.prediction {

    font-size: 30px;

    font-weight: bold;

    margin: 10px;

}


.confidence {

    font-size: 18px;

    color: #555;

}


.images {

    display: flex;

    justify-content: center;

    gap: 20px;

    flex-wrap: wrap;

    margin-top: 30px;

}


.card {

    width: 280px;

}


.card img {

    width: 100%;

    border-radius: 12px;

}


.card h3 {

    margin-bottom: 8px;

}


.note {

    margin-top: 30px;

    font-size: 13px;

    color: #777;

}


.error {

    color: #b91c1c;

    font-weight: bold;

}

</style>

</head>


<body>


<div class="container">

<h1>🧬 BioLens AI</h1>

<div class="subtitle">

Multimodal Disease Risk Prediction from Body Images

</div>


<form
    method="POST"
    enctype="multipart/form-data"
>

<input
    type="file"
    name="image"
    accept="image/*"
    required
>

<br>

<button type="submit">

Analyze Image

</button>

</form>


{% if error %}

<p class="error">

{{ error }}

</p>

{% endif %}


{% if prediction %}

<div class="result">

<div class="prediction">

Prediction: {{ prediction }}

</div>


<div class="confidence">

Confidence:
{{ confidence }}%

</div>


<div class="images">


<div class="card">

<h3>Original Image</h3>

<img
    src="/uploads/original.jpg"
>

</div>


<div class="card">

<h3>ViT Grad-CAM</h3>

<img
    src="/uploads/heatmap.jpg"
>

</div>


<div class="card">

<h3>Explanation Overlay</h3>

<img
    src="/uploads/overlay.jpg"
>

</div>


</div>

</div>

{% endif %}


<div class="note">

Research prototype — not intended for clinical diagnosis.

</div>


</div>


</body>

</html>
"""


# ==================================================
# ROUTE
# ==================================================

@app.route(
    "/",
    methods=["GET", "POST"]
)

def home():

    prediction = None

    confidence = None

    error = None


    if request.method == "POST":

        uploaded_file = request.files.get(
            "image"
        )


        if uploaded_file is None:

            error = "Please select an image."

            return render_template_string(
                HTML,
                prediction=prediction,
                confidence=confidence,
                error=error
            )


        if uploaded_file.filename == "":

            error = "Please select an image."

            return render_template_string(
                HTML,
                prediction=prediction,
                confidence=confidence,
                error=error
            )


        try:

            # --------------------------------------
            # Load uploaded image
            # --------------------------------------

            image = Image.open(
                uploaded_file
            ).convert("RGB")


            # --------------------------------------
            # Prepare input
            # --------------------------------------

            input_tensor = transform(
                image
            ).unsqueeze(0)

            input_tensor = input_tensor.to(
                DEVICE
            )


            # --------------------------------------
            # Prediction
            # --------------------------------------

            with torch.no_grad():

                outputs = model(
                    input_tensor
                )

                probabilities = (
                    torch.softmax(
                        outputs,
                        dim=1
                    )
                )

                prediction_index = (
                    torch.argmax(
                        outputs,
                        dim=1
                    ).item()
                )


            prediction = CLASS_NAMES[
                prediction_index
            ]


            confidence = round(
                probabilities[
                    0,
                    prediction_index
                ].item() * 100,
                2
            )


            # --------------------------------------
            # Generate explanation
            # --------------------------------------

            generate_gradcam(
                image,
                input_tensor,
                prediction_index
            )


        except Exception as e:

            error = (
                "Error processing image: "
                + str(e)
            )


    return render_template_string(
        HTML,
        prediction=prediction,
        confidence=confidence,
        error=error
    )


# ==================================================
# SERVE UPLOADED IMAGES
# ==================================================

@app.route(
    "/uploads/<filename>"
)

def uploaded_file(filename):

    from flask import send_from_directory

    return send_from_directory(
        UPLOAD_FOLDER,
        filename
    )


# ==================================================
# RUN APPLICATION
# ==================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("BioLens AI APPLICATION")
    print("=" * 60)
    print("Open your browser at:")
    print("http://127.0.0.1:5000")
    print("=" * 60)

    app.run(
        debug=False
    )