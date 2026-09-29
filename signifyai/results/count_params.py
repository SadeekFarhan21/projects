"""Rebuilds the three CNNs with the exact layer stacks from model_builders/*.py
(copied by hand, no data needed) and reports parameter counts.
Run: python results/count_params.py > results/param_counts.txt"""
import os, tensorflow as tf
from tensorflow.keras import layers as L
tf.get_logger().setLevel("ERROR")

def gesture(n=10):
    return tf.keras.Sequential([L.InputLayer(input_shape=(128,128,1)), L.Conv2D(16,3,padding='same',activation='relu'),
        L.MaxPooling2D(), L.Flatten(), L.Dense(32,activation='relu'), L.Dense(n,activation='softmax')])
def asl(n):
    return tf.keras.Sequential([L.Conv2D(32,3,activation='relu',input_shape=(128,128,1)), L.MaxPooling2D(2),
        L.Conv2D(64,3,activation='relu'), L.MaxPooling2D(2), L.Conv2D(128,3,activation='relu'), L.MaxPooling2D(2),
        L.Flatten(), L.Dense(128,activation='relu'), L.Dropout(0.5), L.Dense(n,activation='softmax')])
def emotion(n=7):
    m=[L.Input(shape=(48,48,1))]
    for f,d in [(32,.25),(64,.35),(128,.5)]:
        for _ in range(2): m += [L.Conv2D(f,3,padding='same',activation='relu'), L.BatchNormalization()]
        m += [L.MaxPooling2D(2), L.Dropout(d)]
    m += [L.Conv2D(512,3,padding='same',activation='relu'), L.BatchNormalization(), L.Dropout(.5),
          L.GlobalAveragePooling2D(), L.Dense(64,activation='relu'), L.BatchNormalization(), L.Dropout(.5),
          L.Dense(n,activation='softmax')]
    return tf.keras.Sequential(m)
for name, m in [("gesture (10 classes)",gesture()),("asl (29 classes, current code)",asl(29)),
                ("asl (36 classes, paper-era)",asl(36)),("emotion (7 classes)",emotion())]:
    print(f"{name}: total={m.count_params():,} trainable={sum(int(tf.size(w)) for w in m.trainable_weights):,}")
