import os
import gdown
import tensorflow as tf
import streamlit as st
from PIL import Image
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model details
MODEL_PATH = "oxford_flower102_model_trained.h5"
DRIVE_FILE_ID = "1pZ_DJIpa9YXSxB32rASBYw-VuFjgciY2"  # Google Drive file ID

# Function to download model if not available locally
def download_model():
    """Download the model from Google Drive if not present"""
    url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

# Function to load the model
@st.cache_resource
def load_model():
    """Load the model, downloading it first if necessary"""
    if not os.path.exists(MODEL_PATH):
        logger.error("Model file not found at %s", MODEL_PATH)
        st.warning(f"Model file {MODEL_PATH} not found! Downloading from Google Drive...")
        download_model()
    logger.info("Loading model from %s", MODEL_PATH)
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    logger.info("Model loaded successfully")
    return model

# Function to predict the class of the input image
def predict_class(image, model):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, [180, 180])
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    return prediction

# Load the model
model = load_model()

# Streamlit UI
st.title("Flower Classification App")
st.write("Upload an image to classify the flower type.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

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

if uploaded_file is None:
    st.text("Waiting for upload...")
else:
    slot = st.empty()
    slot.text("Running inference...")

    test_image = Image.open(uploaded_file)
    st.image(test_image, caption="Uploaded Image", use_column_width=True)

    pred = predict_class(np.asarray(test_image), model)
    result = class_names[np.argmax(pred)]
    output = "The image is a " + result

    slot.text("Done")
    st.success(output)
