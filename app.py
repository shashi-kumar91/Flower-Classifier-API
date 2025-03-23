from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import requests

app = Flask(__name__)

def load_model():
    model_path = 'oxford_flower102_model_trained.h5'
    if not os.path.exists(model_path):
        print("Downloading model (225 MB, may take a moment)...")
        url = "https://drive.google.com/file/d/1pZ_DJIpa9YXSxB32rASBYw-VuFjgciY2/view?usp=drive_link"
        # Fix URL for direct download
        file_id = url.split('/d/')[1].split('/')[0]  # Extract file ID
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        response = requests.get(download_url)
        with open(model_path, 'wb') as f:
            f.write(response.content)
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

def predict_class(image, model):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, [180, 180])
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    return prediction

model = load_model()

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

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'message': 'Waiting for upload....'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': 'Waiting for upload....'})

    test_image = Image.open(file)
    pred = predict_class(np.asarray(test_image), model)

    result = class_names[np.argmax(pred)]
    output = 'The image is a ' + result

    return jsonify({'message': 'Done', 'prediction': output})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
