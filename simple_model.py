from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Lambda, Cropping2D, Dropout
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import cv2
import csv
import os
import sklearn
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
lines = lines[1:]
images = []
measurements = []
for line in lines:
    source_path = line[0]
    file_name = source_path.split('/')[-1]
    current_path = 'data/IMG/' + file_name
    image = mpimg.imread(current_path)
    measurement = float(line[3])
    images.append(image)
    measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)


ch, row, col = 3, 160, 320  # Trimmed image format

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation
model.add(Lambda(lambda x: x/255.0 - 0.5,
        input_shape=(row, col, ch)))
model.add(Cropping2D(cropping = ((70,25),(0,0))))
model.add(Conv2D(24, 5, strides=(2,2), activation="relu"))
model.add(Conv2D(36, 5, strides=(2,2), activation="relu"))
model.add(Conv2D(48, 5, strides=(2,2), activation="relu"))
model.add(Conv2D(64, 3, activation="relu"))
model.add(Conv2D(64, 3, activation="relu"))
model.add(Flatten())
model.add(Dropout(.2))
model.add(Dense(100))
model.add(Dropout(.2))
model.add(Dense(50))
model.add(Dropout(.5))
model.add(Dense(10))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True)

model.save('model.h5')
