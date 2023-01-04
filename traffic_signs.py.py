import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.utils import to_categorical
from PIL import Image
from sklearn.preprocessing import OneHotEncoder
from tensorflow import keras

# Load the image paths and labels

meta = pd.read_csv('Meta.csv')
num_classes = meta.index.size

train_paths = pd.read_csv('Train.csv')['Path'].values
test_paths = pd.read_csv('Test.csv')['Path'].values

train_labels = np.array([pd.read_csv('Train.csv')['ClassId'].values]).T
test_labels = np.array([pd.read_csv('Test.csv')['ClassId'].values]).T
train_labels = to_categorical(train_labels, num_classes)
test_labels = to_categorical(test_labels, num_classes)

np.random.RandomState(seed=42).shuffle(train_paths)
np.random.RandomState(seed=42).shuffle(train_labels)

# Load the images

images = []
for path in train_paths:
    image = Image.open(path)
    image = image.resize((30, 30))
    image = np.array(image)/255.0
    images.append(image)
train_images = np.array(images)

# Define the model

model = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(30, 30, 3)),
    keras.layers.Conv2D(32, kernel_size=(7, 7), activation='relu'),
    keras.layers.BatchNormalization(axis=3),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.2),
    keras.layers.Conv2D(64, kernel_size=(5, 5), activation='relu'),
    keras.layers.BatchNormalization(axis=3),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.2),
    keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
    keras.layers.BatchNormalization(axis=3),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.3),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(num_classes, activation='linear')
])

# Compile the model

model.compile(loss=keras.losses.CategoricalCrossentropy(
    from_logits=True), optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

# Fit the model

model.fit(train_images, train_labels, epochs=10,
          validation_split=0.2, batch_size=64, verbose=1)

# Save the model

model.save('traffic_signs.h5')

# Load the model

model = keras.models.load_model('traffic_signs.h5')

# Evaluate the model using test data

images = []
for path in test_paths:
    image = Image.open(path)
    image = image.resize((30, 30))
    image = np.array(image)/255.0
    images.append(image)
test_images = np.array(images)

scores = model.evaluate(test_images, test_labels, verbose=1)

print(f"Accuracy on test data: {scores[1]*100}")
