import os
import gdown
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model details
MODEL_PATH = "oxford_flower102_model_trained.h5"
DRIVE_FILE_ID = "1pZ_DJIpa9YXSxB32rASBYw-VuFjgciY2"

# Function to download model if not available locally
def download_model():
    """Download the model from Google Drive if not present"""
    url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
    logger.info("Downloading model from Google Drive...")
    try:
        gdown.download(url, MODEL_PATH, quiet=False)
        logger.info("Model downloaded successfully")
    except Exception as e:
        logger.error("Failed to download model: %s", str(e))
        raise Exception(f"Model download failed: {str(e)}")

# Initialize FastAPI app
app = FastAPI(title="Flower Classifier API")

# Mount static files
app.mount("/", StaticFiles(directory="static", html=True), name="static")

# Global model variable
model = None

# Load model at startup
@app.on_event("startup")
async def startup_event():
    global model
    if not os.path.exists(MODEL_PATH):
        logger.error("Model file not found at %s", MODEL_PATH)
        download_model()
    logger.info("Loading model from %s", MODEL_PATH)
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error("Failed to load model: %s", str(e))
        raise Exception(f"Model loading failed: {str(e)}")

# Function to predict the class of the input image
def predict_class(image: Image.Image, model):
    image = tf.cast(np.asarray(image), tf.float32)
    image = tf.image.resize(image, [180, 180])
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    return "Prediction Result"

# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        logger.info("Image uploaded and opened successfully")
        result = predict_class(image, model)
        logger.info("Prediction: %s", result)
        return {"prediction": result}
    except Exception as e:
        logger.error("Error processing image: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health():
    return {"message": "Flower Classifier API is running"}

# Run the app with dynamic port binding
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))  # Use PORT env var, default to 8000 locally
    uvicorn.run(app, host="0.0.0.0", port=port)
