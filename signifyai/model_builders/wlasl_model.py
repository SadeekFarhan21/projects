import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np

# Fix BASE_DIR to point to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "asl_dataset")

# Load datasets with correct splits
batch_size = 32
img_size = (128, 128)

train_ds = image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size,
    color_mode='grayscale',
    label_mode='int'
)

val_ds = image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",   
    seed=123,            
    image_size=img_size,
    batch_size=batch_size,
    color_mode='grayscale',
    label_mode='int'
)

# No separate test split - we'll use validation set for testing
test_ds = val_ds

# Save class names before preprocessing (which strips the attribute)
class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Classes ({num_classes}): {class_names}")

# Add preprocessing utility function to normalize data consistently
def preprocess_data(ds):
    # Rescale data consistently
    normalization_layer = layers.Rescaling(1./255)
    return ds.map(lambda x, y: (normalization_layer(x), y))

# Normalize datasets
train_ds = preprocess_data(train_ds)
val_ds = preprocess_data(val_ds)
test_ds = preprocess_data(test_ds)

# Verify preprocessing
for images, labels in train_ds.take(1):
    print("Training data range:", np.min(images.numpy()), "to", np.max(images.numpy()))

# Print dataset info
print("Number of training batches:", len(train_ds))
print("Number of validation batches:", len(val_ds))

# Define the CNN model with single channel input - removed rescaling layer
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
epochs = 30
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

# Evaluate model
model.evaluate(test_ds)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('WLASL Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()

# Save model to correct path relative to project root
model_save_path = os.path.join(BASE_DIR, "app", "models", "asl", "asl_model.keras")
model.save(model_save_path)
print(f"Model saved to: {model_save_path}")