import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Load the Oxford Flowers-102 dataset
dataset_name = "oxford_flowers102"
ds_info = tfds.builder(dataset_name).info
num_classes = ds_info.features["label"].num_classes

# Define image size and batch size
image_size = (224, 224)
batch_size = 32

# Load and preprocess the data
train_data, validation_data, test_data = tfds.load(dataset_name, split=["train[:80%]", "train[80%:90%]", "train[90%:]"], as_supervised=True)

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

# Data Augmentation
def preprocess_image(image, label):
    image = tf.image.resize(image, image_size)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    return image, label

train_data = train_data.map(preprocess_image).shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
validation_data = validation_data.map(preprocess_image).batch(batch_size).prefetch(tf.data.AUTOTUNE)

def preprocess_image1(image, label):
    image = tf.image.resize(image, image_size)
    return image, label

test_data = test_data.map(preprocess_image1).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Get a single batch from the train data (moved after batching)
train_batch = next(iter(train_data))
train_image, train_label = train_batch

# Visualization of flowers from dataset
plt.imshow(train_image[10].numpy().astype("uint8"))
plt.title(f'Label: {class_names[train_label[10]]}')
plt.axis('off')
plt.show()

plt.imshow(train_image[9].numpy().astype("uint8"))
plt.title(f'Label: {class_names[train_label[9]]}')
plt.axis('off')
plt.show()

# Load ResNet50 base model
base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False)

# Fine-tune last few layers of the base model
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Visualizing layers of ResNet50
base_model.summary()

for num, layer in enumerate(base_model.layers):
    print(num, layer.name, layer.trainable)

# Add custom head for fine-tuning (Phase 1)
inputs = tf.keras.layers.Input(shape=(224, 224, 3))
x = base_model(inputs)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(102, activation='softmax')(x)

model = tf.keras.models.Model(inputs, outputs)

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

lr = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4*10**(epoch/20))

history = model.fit(
    train_data,
    epochs=20,
    validation_data=validation_data,
    callbacks=[lr]
)

# Add custom head for fine-tuning (Phase 2)
inputs = tf.keras.layers.Input(shape=(224, 224, 3))
x = base_model(inputs)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(102, activation='softmax')(x)

model = tf.keras.models.Model(inputs, outputs)

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(learning_rate=2.8184e-04),
              metrics=['accuracy'])

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

history = model.fit(
    train_data,
    epochs=10,
    validation_data=validation_data,
    callbacks=[early_stopping]
)

# Get a single batch from the test data
test_batch = next(iter(test_data))
test_image, test_label = test_batch

# Make predictions on the test image using the trained model
predictions = model.predict(test_image)

# Get the index of the predicted label with the highest probability
predicted_index = np.argmax(predictions[18])

# Get the corresponding class name from the class_names list
predicted_label = class_names[predicted_index]
plt.imshow(test_image[18].numpy().astype("uint8"))
plt.title(f'Predicted Label: {predicted_label}')
plt.axis('off')
plt.show()

# --- Visualization ---
# Plot Training History (Accuracy and Loss)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig('training_history.png')  # Save instead of show

# Get predictions for the test dataset
y_pred = model.predict(test_data)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.concatenate([y for x, y in test_data], axis=0)

# Create and plot confusion matrix
confusion_mtx = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(15, 10))
sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')  # Save instead of show

# Randomly select some images from the test dataset
test_data_list = list(test_data)
sample_indices = np.random.choice(len(test_data_list), size=min(9, len(test_data_list)), replace=False)  # Adjusted size

plt.figure(figsize=(12, 12))
for i, index in enumerate(sample_indices):
    plt.subplot(3, 3, i + 1)
    image, label = test_data_list[index][0][0], test_data_list[index][1][0]
    plt.imshow(image.numpy().astype("uint8"))
    predicted_label = np.argmax(model.predict(tf.expand_dims(image, 0)))
    plt.title(f'True: {class_names[label]}\nPredicted: {class_names[predicted_label]}')
    plt.axis('off')

plt.savefig('test_samples.png')  # Save instead of show

# Save the model
model.save('oxford_flower102_model_trained.h5')
print("Model saved as 'oxford_flower102_model_trained.h5'")