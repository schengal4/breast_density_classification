import os
import io
import torch
import base64
import numpy as np
import tempfile
from typing import List, Union

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse

from PIL import Image

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    Resized
)
from monai.networks.nets import TorchVisionFCModel

############################
# GLOBALS / INITIALIZATION #
############################

# 4 breast density categories (or adjust if your labeling differs)
BREAST_CLASSES = ["A", "B", "C", "D"]

# Build a MONAI Compose for image preprocessing
preprocess = Compose([
    # We rely on saving the file to disk and passing that path to `LoadImaged`.
    LoadImaged(keys="image", image_only=False),  # or `image_only=True`, both can work
    EnsureChannelFirstd(keys="image", channel_dim=2),  # shape => [C, H, W]
    ScaleIntensityd(keys="image", minv=0.0, maxv=1.0),
    Resized(keys="image", spatial_size=(299, 299))
])

# Choose CPU or GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(checkpoint_path: str = "models/model.pt") -> TorchVisionFCModel:
    """
    Create the same model architecture used in training and load the checkpoint.
    """
    model = TorchVisionFCModel(
        model_name="inception_v3",
        num_classes=4,
        pretrained=True,  # or False, depending on how you trained
        pool=None,
        bias=True,
        use_conv=False
    ).to(DEVICE)

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model

# Initialize the FastAPI app
app = FastAPI(title="Breast Density Classification API")

# Load the model once at startup
MODEL = load_model("models/model.pt")  # adapt path as needed

##################
# HELPER METHODS #
##################

def validate_image_extension(filename: str) -> None:
    """
    Raise HTTPException if file extension isn't recognized.
    """
    valid_exts = [".jpg", ".jpeg", ".png", ".dcm"]
    ext = os.path.splitext(filename)[1].lower()
    if ext not in valid_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension '{ext}'. Must be one of {valid_exts}."
        )

def run_inference(image_bytes: Union[bytes, None] = None, example: bool = False) -> dict:
    """
    1) Preprocess the image (Load, channel-first, scale, resize).
    2) Run model inference -> return predicted class and probabilities.
    If `example=True`, we skip the file upload bytes and load a known sample path instead.
    """
    if not example:
        # Save uploaded bytes to a temp file so LoadImaged can read from disk
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(image_bytes)
            tmp.flush()
            tmp_path = tmp.name
        data_dict = {"image": tmp_path}
    else:
        # Hardcode a known sample path, e.g. "sample_data/A/sample_A1.jpg"
        data_dict = {"image": "sample_data/A/sample_A1.jpg"}

    # 1) Apply the MONAI Compose pipeline
    out_dict = preprocess(data_dict)
    # out_dict["image"] is now a NumPy array or MetaTensor
    image_np = out_dict["image"]

    # 2) Convert to PyTorch tensor & add batch dimension => shape [1, C, H, W]
    tensor = torch.as_tensor(image_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # 3) Model inference
    with torch.no_grad():
        logits = MODEL(tensor)   # shape => [1, 4]
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]  # shape => [4]

    # 4) Determine predicted class
    predicted_idx = np.argmax(probs)
    predicted_class = BREAST_CLASSES[predicted_idx]

    # 5) Build response
    result = {
        "predicted_class": predicted_class,
        "probabilities": {
            BREAST_CLASSES[i]: float(probs[i]) for i in range(len(probs))
        }
    }
    return result

##################
# API ENDPOINTS  #
##################

@app.get("/")
def homepage():
    """
    A simple homepage that references the Hugging Face model.
    """
    html_content = """
    <html>
    <head>
        <title>Breast Density Classification API</title>
    </head>
    <body>
        <h1>Welcome to the Breast Density Classification API</h1>
        <p>
          This API classifies mammogram images into one of four density categories (A, B, C, or D).
          The underlying model is a pre-trained InceptionV3-based classifier from
          <a href="https://huggingface.co/monai-test/breast_density_classification">
            monai-test/breast_density_classification
          </a> on Hugging Face.
        </p>
        <p>Available endpoints:</p>
        <ul>
          <li><strong>POST /classify</strong> - Upload an image file (JPEG, PNG, DICOM) to get the predicted class & probabilities.</li>
          <li><strong>POST /example</strong> - Returns a sample image's classification & the sample image (base64) as an example.</li>
        </ul>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)

@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    """
    POST an image file (JPEG, PNG, DICOM, etc.) to get
    the predicted breast density class + probabilities.
    """
    # 1) Validate extension
    validate_image_extension(file.filename)

    # 2) Read file contents
    image_bytes = await file.read()

    # 3) Run inference
    try:
        results = run_inference(image_bytes, example=False)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during inference: {str(e)}"
        )

    # 4) Return JSON
    return JSONResponse(content=results)

@app.post("/example")
async def classify_example_image():
    """
    Returns classification results for a known example image,
    along with the example image in base64 form.
    """
    try:
        # 1) Run inference in 'example' mode
        results = run_inference(example=True)

        # 2) Also return the example image base64-encoded
        #    So the user can see exactly which image was used
        example_image_path = "sample_data/A/sample_A1.jpg"
        if not os.path.exists(example_image_path):
            raise HTTPException(
                status_code=404,
                detail=f"Sample image '{example_image_path}' not found."
            )

        with open(example_image_path, "rb") as f:
            image_data = f.read()
        # Convert to base64 string
        b64_str = base64.b64encode(image_data).decode("utf-8")

        combined_response = {
            "inference": results,
            "example_image_b64": b64_str
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during inference on example image: {str(e)}"
        )

    return JSONResponse(content=combined_response)
