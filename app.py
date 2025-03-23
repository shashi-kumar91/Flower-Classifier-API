import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to load the pre-trained model and cache it
@st.cache_resource
def load_model():
    model_path = 'oxford_flower102_model_trained.h5'  # Assumes model is in root directory
    if not os.path.exists(model_path):
        logger.error("Model file not found at %s", model_path)
        raise FileNotFoundError(f"Model file {model_path} not found!")
    logger.info("Loading model from %s", model_path)
    model = tf.keras.models.load_model(model_path, compile=False)
    logger.info("Model loaded successfully")
    return model

# Function to predict the class of the input image
def predict_class(image, model):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, [180, 180])
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    return prediction

# Load the model (cached)
model = load_model()

# Streamlit interface
st.title("Flower Classifier")

file = st.file_uploader("Upload a flower image", type=["jpg", "jpeg", "png"])

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

if file is None:
    st.text("Waiting for upload...")
else:
    slot = st.empty()
    slot.text("Running inference...")

    test_image = Image.open(file)
    st.image(test_image, caption="Input Image", width=400)

    pred = predict_class(np.asarray(test_image), model)
    result = class_names[np.argmax(pred)]
    output = "The image is a " + result

    slot.text("Done")
    st.success(output)
