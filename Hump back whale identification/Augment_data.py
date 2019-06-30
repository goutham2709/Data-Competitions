import numpy as np
from numpy.random import normal, random
from scipy.ndimage.interpolation import rotate, shift

def random_flip(image, p=0.5):
    if random() < p:
        image = np.fliplr(image)
    return image

def random_rotate(image, sigma=10):
    angle = sigma * normal()
    return rotate(image, angle, reshape=False)

def random_translate(image, sigma=3):
    horizontal_shift = sigma * normal()
    vertical_shift = sigma * normal()   
    image = shift(image, [vertical_shift, horizontal_shift,0])
    return image

def random_zoom(image):
#     TODO: Implement this?
    return image

def random_transform(image):
    image = random_flip(image)    
    image = random_translate(image)
    image = random_rotate(image)
#     image = random_zoom(image)
    return image