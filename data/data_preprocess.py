import numpy as np


def normalize(x, mean, std):
    # normalize each training example by subtracting the mean and dividing by std
    return ((x - mean) / std).astype('float32')


def pad(x, pixels_to_pad):
    """
    if x_train.shape = (50000, 32, 32, 3) Results in x_train.shape = (50000, 40, 40, 3)
    Define a function that pads pixels_to_pad pixels along axis 1 and 2
    and Pads with the reflection of the vector mirrored on the first and last values of the vector along each axis.
    """
    return np.pad(x, [(0, 0), (pixels_to_pad, pixels_to_pad), (pixels_to_pad, pixels_to_pad), (0, 0)],
                  mode='reflect')
