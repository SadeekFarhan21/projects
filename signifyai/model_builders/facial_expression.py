# %%
import keras
import tensorflow as tf
import math

# %%
import numpy as np  # linear algebra
import pandas as pd  # df processing, CSV file I/O (e.g. pd.read_csv)
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Update CSV path using os.path.join
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# %%
df = pd.read_csv(os.path.join(BASE_DIR, "data", "fer2013.csv"))

# %%
df.head()

# %%
df.Usage.value_counts()

# %%
emotion_map = {0: 'Angry', 1: 'Digust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
emotion_counts = df['emotion'].value_counts(sort=False).reset_index()
emotion_counts.columns = ['emotion', 'number']
emotion_counts['emotion'] = emotion_counts['emotion'].map(emotion_map)
emotion_counts

# %%
def row2image(row):
    pixels, emotion = row['pixels'], emotion_map[row['emotion']]
    img = np.array(pixels.split(), dtype=int)
    img = img.reshape(48,48)
    image = np.zeros((48,48,3))
    image[:,:,0] = img
    image[:,:,1] = img
    image[:,:,2] = img
    return image.astype(np.uint8), emotion

# %%
data_train = df[df['Usage']=='Training'].copy()
data_val   = df[df['Usage']=='PublicTest'].copy()
data_test  = df[df['Usage']=='PrivateTest'].copy()
print("train shape: {}, \nvalidation shape: {}, \ntest shape: {}".format(data_train.shape, data_val.shape, data_test.shape))

# %%
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# %%
num_classes = 7 
width, height = 48, 48
num_epochs = 100
batch_size = 64
num_features = 64

# New helper function to parse pixels
def parse_pixels(pixel_str: str) -> list:
    return [int(token) for token in pixel_str.split()]

def CRNO(df: pd.DataFrame, dataName: str):
    df['pixels'] = df['pixels'].apply(parse_pixels)
    data_X = np.array(df['pixels'].tolist(), dtype='float32').reshape(-1, width, height, 1) / 255.0
    data_Y = to_categorical(df['emotion'], num_classes)
    print(f"{dataName}_X shape: {data_X.shape}, {dataName}_Y shape: {data_Y.shape}")
    return data_X, data_Y

def main():
    train_X, train_Y = CRNO(data_train, "train")   # training data
    val_X, val_Y     = CRNO(data_val, "val")         # validation data
    test_X, test_Y   = CRNO(data_test, "test")        # test data

    # Updated model architecture with an additional convolutional block for improved accuracy
    
    model = Sequential([
        tf.keras.layers.Input(shape=(width, height, 1)),
        # Block 1
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Block 2
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.35),
        
        # Block 3
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.5),
        
        # Block 4
        Conv2D(512, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        
        # Global Pooling and additional small dense layer
        tf.keras.layers.GlobalAveragePooling2D(),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', 
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=['accuracy'])
    model.summary()

    # Set up data augmentation
    data_generator = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )

    es = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)
    rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    history = model.fit(
        data_generator.flow(train_X, train_Y, batch_size),
        epochs=num_epochs,
        verbose=1,
        callbacks=[es, rlrop],
        validation_data=(val_X, val_Y)
    )

    test_loss, test_accuracy = model.evaluate(test_X, test_Y, verbose=1)
    print(f"Test loss: {test_loss}")
    print(f"Test accuracy: {test_accuracy}")

    # Update save path to go up one directory then into app/models/emotion
    parent_dir = os.path.dirname(BASE_DIR)  # Go up one level from model_builders
    model_save_path = os.path.join(parent_dir, "app", "models", "emotion", "emotion.keras")
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    model.save(model_save_path)
    print(f"Model saved to: {model_save_path}")

if __name__ == '__main__':
    main()