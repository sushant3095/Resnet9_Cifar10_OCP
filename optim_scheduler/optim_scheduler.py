from config import LEARNING_RATE, EPOCHS, BATCH_SIZE, MOMENTUM
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


class OptSchedule():
    def plot_lr_schedule(self):
        x = list()
        y = list()
        for i in range(EPOCHS):
            x.append(i)
            y.append(self.lr_schedule(i))

        plt.xlabel('Epochs')
        plt.ylabel('Learning Rate')
        plt.title('Learning-Rate Schedule')
        plt.plot(x, y)

    def __init__(self, batches_per_epoch):
        self.lr_schedule = lambda t: np.interp([t], [0, (EPOCHS+1)//5, EPOCHS], [0, LEARNING_RATE, 0])[0]
        self.global_step = tf.train.get_or_create_global_step()
        self.lr_func = lambda: self.lr_schedule(self.global_step/batches_per_epoch)/BATCH_SIZE
        self.opt = tf.train.MomentumOptimizer(self.lr_func, momentum=MOMENTUM, use_nesterov=True)