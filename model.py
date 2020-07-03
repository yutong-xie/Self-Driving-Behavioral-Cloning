from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Lambda, Cropping2D, Dropout

import numpy as np
import tensorflow as tf
import cv2
import csv
import os
import sklearn
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from data_generator import *

samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

samples = samples[1:]

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# Set our batch size
batch_size = 32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

ch, row, col = 3, 66, 200  # Trimmed image format

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation
model.add(Lambda(lambda x: x/255.0 - 0.5,
        input_shape=(row, col, ch)))
model.add(Conv2D(24, 5, strides=(2,2), activation="elu"))
model.add(Conv2D(36, 5, strides=(2,2), activation="elu"))
model.add(Conv2D(48, 5, strides=(2,2), activation="elu"))
model.add(Conv2D(64, 3, activation="elu"))
model.add(Conv2D(64, 3, activation="elu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(.2))
model.add(Dense(50))
model.add(Dropout(.2))
model.add(Dense(10))
model.add(Dropout(.5))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator,
            samples_per_epoch = len(train_samples),
            validation_data = validation_generator,
            nb_val_samples = len(validation_samples),
            nb_epoch=10, verbose=1)

model.save('model.h5')
