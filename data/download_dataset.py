import tensorflow as tf
import numpy as np

from config import TF_RECORD_TRAIN_FILEPATH, TF_RECORD_TEST_FILEPATH
from data.data_preprocess import normalize, pad
from data.convert_to_tf_records import write_tf_record


class InputPipeline():
    def download_dataset(self):
        # Download the The CIFAR-10 data-set
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        return x_train, y_train, x_test, y_test

    def compute_batches_per_epoch(self, batch_size):
        return self.len_train // batch_size + 1

    def __init__(self):
        print("Downloading CIFAR-10 data-set")
        self.x_train, self.y_train, self.x_test, self.y_test = self.download_dataset()

        # len_train: Number of training examples,
        # len_test: Number of test examples in the data-set
        self.len_train = len(self.x_train)
        self.len_test = len(self.x_test)

        self.x_train = self.x_train.astype('float32')
        self.x_test = self.x_test.astype('float32')

        # Compute Mean and STD on Training Set
        self.train_mean = np.mean(self.x_train, axis=(0, 1, 2))
        self.train_std = np.std(self.x_train, axis=(0, 1, 2))

        # Reshape into 1-D Arrays
        self.train_labels = self.y_train.astype('int64').reshape(self.len_train)
        self.test_labels = self.y_test.astype('int64').reshape(self.len_test)

        # Pre-process x_train by adding padding -> Normalize
        print("Pre-process x_train by: adding padding -> Normalize")
        self.train_features = normalize(pad(self.x_train, 4), self.train_mean, self.train_std)

        # Pre-process x_test by Normalize
        print("Pre-process x_test by: Normalize")
        self.test_features = normalize(self.x_test, self.train_mean, self.train_std)

        print("Writing train_features to : ", TF_RECORD_TRAIN_FILEPATH)
        write_tf_record(self.train_features, self.train_labels, TF_RECORD_TRAIN_FILEPATH)
        print("Writing test_features to : ", TF_RECORD_TEST_FILEPATH)
        write_tf_record(self.test_features, self.test_labels, TF_RECORD_TEST_FILEPATH)

        print("Done")






