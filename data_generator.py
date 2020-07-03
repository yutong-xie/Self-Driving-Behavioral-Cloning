import numpy as np
import cv2
import csv
import os
import sklearn
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import scipy.ndimage
import random

def crop_image(image):
    '''
    Crop the image, remove sky and the front of car
    image_size: 160x320x3 -> 65x320x3
    '''
    return image[70:-25, :, :]

def resize_image(image):
    '''
    Resize image to 66x200x3 which is the input size of Nvidia network
    Interpolation method is INTER_AREA, which is preferred method for image decimation
    '''
    return cv2.resize(image, (200, 66), interpolation = cv2.INTER_AREA)


def random_flip(image, angle):
    '''
    Randomly flip the image.
    '''
    random_num = random.random()
    if random_num > 0.5:
        return np.fliplr(image), -1.0*angle
    else:
        return image, angle

def random_brightness(image):
    '''
    Change image to HSV color format and randomly change the brightness
    '''
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 0.5 + 0.5*random.random()
    image[:,:,2] = image[:,:,2]*ratio
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    return image

def random_rotation(image, angle, rotation):
    '''
    Randomly rotate the image and get new angles
    '''
    random_num = random.random()
    if random_num > 0.5:
        random_rotation = random.randrange(-rotation, rotation)
        rad = random_rotation / 180 * np.pi
        image = scipy.ndimage.rotate(image, random_rotation, reshape = False)
        return image, angle + (-1) * rad
    else:
        return image, angle

def data_augmentation(image, angle):
    image = crop_image(image)
    image = resize_image(image)
    image = random_brightness(image)
    image, angle = random_flip(image, angle)
    image, angle = random_rotation(image, angle, 15)

    return image, angle

def generator(samples, batch_size=32, valid_flag=False):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        # sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                correction = [0, 0.25, -0.25]
                # Randomly select left or right or center image
                index = random.randrange(0,3)

                name = 'data/IMG/'+batch_sample[index].split('/')[-1]

                image = mpimg.imread(name)

                angle = float(batch_sample[3]) + correction[index]

                image, angle = data_augmentation(image, angle)

                images.append(image)
                angles.append(angle)


            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
