import tensorflow as tf
import os
from tensorflow.keras import layers, models, Sequential
from tensorflow.data import AUTOTUNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import cv2

# Define parameters
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join("data", "leapGestRecog")
IMG_SIZE = (128, 128)  # Modified
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.0005

def load_dataset(dataset_path):
    images, labels = [], []
    label_map = {}
    label_index = 0
    
    # Simple dataset loading without balancing
    for subject in sorted(os.listdir(dataset_path)):
        subject_path = os.path.join(dataset_path, subject)
        if os.path.isdir(subject_path) and subject.isdigit():
            for gesture in os.listdir(subject_path):
                gesture_path = os.path.join(subject_path, gesture)
                if gesture not in label_map:
                    label_map[gesture] = label_index
                    label_index += 1
                if os.path.isdir(gesture_path):
                    for img_file in os.listdir(gesture_path):
                        img_path = os.path.join(gesture_path, img_file)
                        images.append(img_path)
                        labels.append(label_map[gesture])
    
    return np.array(images), np.array(labels), label_map

# Load dataset
image_paths, labels, label_map = load_dataset(DATASET_PATH)
train_paths, val_paths, train_labels, val_labels = train_test_split(image_paths, labels, test_size=0.3, stratify=labels, random_state=42)
train_paths, test_paths, train_labels, test_labels = train_test_split(train_paths, train_labels, test_size=0.2, stratify=train_labels, random_state=42)

def decode_img(img_path, label):
    # Read image
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=1)  # Read as grayscale
    
    # Simple resize
    img = tf.image.resize(img, IMG_SIZE)
    
    # Normalize to [0,1]
    img = tf.cast(img, tf.float32) / 255.0
    
    return img, label

# Convert dataset to TensorFlow format
train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_ds = train_ds.map(decode_img, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.batch(BATCH_SIZE).shuffle(1000).prefetch(AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
val_ds = val_ds.map(decode_img, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
test_ds = test_ds.map(decode_img, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

print(f"TensorFlow version: {tf.__version__}")

# Create model with explicit names for each layer


model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(128, 128, 1), name='input_1'),
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu', name='conv_1'),
    tf.keras.layers.MaxPooling2D(name='pool_1'),
    tf.keras.layers.Flatten(name='flatten_1'),
    tf.keras.layers.Dense(32, activation='relu', name='dense_1'),
    tf.keras.layers.Dense(len(label_map), activation='softmax', name='output')
])

# Print model summary
model.summary()

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model and save history
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1
)

# Save model with explicit path creation
model_dir = os.path.join(BASE_DIR, "app", "models", "gestures")
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "gesture_model.keras")
abs_model_path = os.path.abspath(model_path)
print(f"\nSaving model to: {abs_model_path}")

model.save(model_path, include_optimizer=False)


# Plot training history
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Gesture Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()


# Verify save
if os.path.exists(model_path):
    print(f"Model saved successfully! File size: {os.path.getsize(model_path) / 1024:.1f}KB")
    # Test load
    try:
        test_model = tf.keras.models.load_model(model_path)
        print("Model test load successful!")
    except Exception as e:
        print(f"Error testing model load: {str(e)}")
else:
    print("Error: Model file not found after saving!")


# Evaluate on test set
print("\nEvaluating on test set:")
test_results = model.evaluate(test_ds, verbose=1)
print(f"Test accuracy: {test_results[1]:.4f}")