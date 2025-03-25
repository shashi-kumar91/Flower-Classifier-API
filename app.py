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
    url = f"https://drive.google.com/uc?export=download&id={DRIVE_FILE_ID}"
    logger.info("Downloading model from Google Drive...")
    try:
        gdown.download(url, MODEL_PATH, quiet=False)
        logger.info("Model downloaded successfully")
    except Exception as e:
        logger.error("Failed to download model: %s", str(e))
        raise

# Initialize FastAPI app
app = FastAPI(title="Flower Classifier API")

# Mount static files
app.mount("/", StaticFiles(directory="static", html=True), name="static")

# Global model variable
model = None

# Class names
class_names = [
    'pink primrose', 'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea',
    'english marigold', 'tiger lily', 'moon orchid', 'bird of paradise', 'monkshood',
    'globe thistle', 'snapdragon', "colt's foot", 'king protea', 'spear thistle',
    'yellow iris', 'globe-flower', 'purple coneflower', 'peruvian lily', 'balloon flower',
    'giant white arum lily', 'fire lily', 'pincushion flower', 'fritillary', 'red ginger',
    'grape hyacinth', 'corn poppy', 'prince of wales feathers', 'stemless gentian', 'artichoke',
    'sweet william', 'carnation', 'garden phlox', 'love in the mist', 'mexican aster',
    'alpine sea holly', 'ruby-lipped cattleya', 'cape flower', 'great masterwort', 'siam tulip',
    'lenten rose', 'barberton daisy', 'daffodil', 'sword lily', 'poinsettia', 'bolero deep blue',
    'wallflower', 'marigold', 'buttercup', 'oxeye daisy', 'common dandelion', 'petunia',
    'wild pansy', 'primula', 'sunflower', 'pelargonium', 'bishop of llandaff', 'gaura', 'geranium',
    'orange dahlia', 'pink-yellow dahlia', 'cautleya spicata', 'japanese anemone', 'black-eyed susan',
    'silverbush', 'californian poppy', 'osteospermum', 'spring crocus', 'bearded iris', 'windflower',
    'tree poppy', 'gazania', 'azalea', 'water lily', 'rose', 'thorn apple', 'morning glory',
    'passion flower', 'lotus lotus', 'toad lily', 'anthurium', 'frangipani', 'clematis', 'hibiscus',
    'columbine', 'desert-rose', 'tree mallow', 'magnolia', 'cyclamen', 'watercress', 'canna lily',
    'hippeastrum', 'bee balm', 'ball moss', 'foxglove', 'bougainvillea', 'camellia', 'mallow',
    'mexican petunia', 'bromelia', 'blanket flower', 'trumpet creeper', 'blackberry lily'
]

# Load model at startup
@app.on_event("startup")
async def startup_event():
    global model
    if not os.path.exists(MODEL_PATH):
        logger.error("Model file not found at %s", MODEL_PATH)
        download_model()
    logger.info("Loading model from %s", MODEL_PATH)
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    logger.info("Model loaded successfully")

# Function to predict the class of the input image
def predict_class(image: Image.Image, model):
    image = tf.cast(np.asarray(image), tf.float32)
    image = tf.image.resize(image, [180, 180])
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    return class_names[np.argmax(prediction)]

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
