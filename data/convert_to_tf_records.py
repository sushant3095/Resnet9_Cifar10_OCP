import tensorflow as tf


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def create_tf_example(image, label):
    return tf.train.Example(
        features=tf.train.Features(feature={'image': _bytes_feature(image.tobytes()), 'label': _int64_feature(label)}))


def write_tf_record(x, y, file_path):
    writer = tf.io.TFRecordWriter(file_path)
    for i, element in enumerate(x):
        tf_example = create_tf_example(element, y[i])
        writer.write(tf_example.SerializeToString())


def decode_tf_record(serialized_example):
    """
    Parses an image and label from the given `serialized_example`.
    It is used as a map function for `dataset.map`
    """
    IMAGE_SIZE = 40
    IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE * 3

    # 1. define a parser
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })

    # 2. Convert the data
    image = tf.decode_raw(features['image'], tf.float32)
    image.set_shape(IMAGE_PIXELS)
    image = tf.cast(tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, 3]), tf.float32)
    label = tf.cast(features['label'], tf.int64)

    # 3. reshape
    return image, label


def decode_test_tf_record(serialized_example):
    """
    Parses an image and label from the given `serialized_example`.
    It is used as a map function for `dataset.map`
    """
    IMAGE_SIZE = 32
    IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE * 3

    # 1. define a parser
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })

    # 2. Convert the data
    image = tf.decode_raw(features['image'], tf.float32)
    image.set_shape(IMAGE_PIXELS)
    image = tf.cast(tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, 3]), tf.float32)
    label = tf.cast(features['label'], tf.int64)

    # 3. reshape
    return image, label
