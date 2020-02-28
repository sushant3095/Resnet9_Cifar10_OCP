import numpy as np
import tensorflow as tf
from config import CUTOUT_PIXELS, RANDOM_CROP_DIMS


def horizontal_flip(image):
    return tf.image.random_flip_left_right(image)


def random_crop(image, dimensions):
    return tf.image.random_crop(image, dimensions)


# CUT-OUT HELPER
def cutout_helper(image, cutout_pixels):
    img_height_width = image.shape[0]

    # randomly choose x and y point on axis
    rand_x = np.random.randint(0, img_height_width + cutout_pixels)
    rand_y = np.random.randint(0, img_height_width + cutout_pixels)

    img_mask = np.ones([img_height_width + cutout_pixels, img_height_width + cutout_pixels, 3])

    # create a black patch on the mask to represent cutoutbox
    img_mask[rand_x: rand_x + cutout_pixels, rand_y: rand_y + cutout_pixels, :] = 0

    # recreate the 32x32 size from that array
    img_mask = img_mask[cutout_pixels: cutout_pixels + img_height_width,
               cutout_pixels: cutout_pixels + img_height_width, :]

    # we need a patch of mean
    # compute mean of the image
    mean_value = tf.reduce_mean(image, axis=(0, 1))
    mean_image = 1 - img_mask
    mean_image = mean_image * mean_value

    # convert both to tensors
    mean_image = tf.convert_to_tensor(mean_image, dtype='float32')
    new_image = tf.convert_to_tensor(img_mask, dtype='float32')

    # patch has value 0, so all other pixels remains same except the patch
    image = image * new_image
    # patch has mean of image, so mean gets added to the patch
    image = image + mean_image

    return image


data_aug = lambda x, y: (cutout_helper(horizontal_flip(random_crop(x, RANDOM_CROP_DIMS)), CUTOUT_PIXELS), y)
