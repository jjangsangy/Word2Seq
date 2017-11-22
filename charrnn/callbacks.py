# -*- coding: utf-8 -*-
"""
Custom keras callbacks modules
"""
from __future__ import print_function

import h5py
import numpy as np

from builtins import super
from keras.callbacks import ModelCheckpoint, Callback
from keras import backend as K


__all__ = 'ModelCheckpoint', 'AdvancedLRScheduler'


class CharRNNCheckpoint(ModelCheckpoint):
    """
    Save checkpoints as well as char-rnn configurations
    """

    def __init__(self, filepath, window, **kwargs):
        """
        filepath: hdf5 file to save weights and configurations
        window: window size to save to file
        """
        self.window = window
        super().__init__(filepath, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs=logs)
        with h5py.File(self.filepath) as h5file:
            h5file.attrs['window'] = self.window


class AdvancedLRScheduler(Callback):
    '''
    Schedule learning rate when a monitored quantity does not
    improve over a period of time or at a particular frequency.
    '''

    def __init__(self, monitor='val_loss', cooldown=0,
                 verbose=False, mode='auto', factor=0.5,
                 min_lr=0, frequency=None):
        """
        monitor: Quantity to be monitored.

        cooldown: Number of epochs with no improvement
                  after which lr will be lowered.

        verbose: Verbosity mode.

        mode: One of {auto, min, max}. In 'min' mode,
              training will stop when the quantity
              monitored has stopped decreasing; in 'max'
              mode it will stop when the quantity
              monitored has stopped increasing.

        factor: Magnitude in which to learning rate decays

        min_lr: Lower bound on the learning rate.

        frequency: Schedule learning rate reduction frequency.
                   Set this to None to disable
        """
        super().__init__()

        self.monitor = monitor
        self.cooldown = cooldown
        self.verbose = verbose
        self.factor = factor
        self.min_lr = min_lr
        self.frequency = frequency
        self.wait = 0

        assert mode in {'auto', 'min', 'max'}

        if mode == 'auto':
            mode = 'max' if 'acc' in monitor else 'min'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf

        if mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf

    def increment(self):
        self.wait += 1

    def reset(self):
        self.wait = 0

    def reduce_lr(self, lr):
        # As long as we are not inside a cooldown
        if self.wait >= self.cooldown:
            lr *= self.factor
            K.set_value(self.model.optimizer.lr, max(lr, self.min_lr))
            if self.verbose:
                print('\nReducing learning rate', end='')
            self.reset()

    def reduce_on_frequency(self, epoch):
        if not self.frequency:
            return False
        return not epoch % self.frequency

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        current = logs.get(self.monitor)
        lr = K.get_value(self.model.optimizer.lr)

        # LR did not improve or frequency
        if not self.monitor_op(current, self.best):
            self.reduce_lr(lr)
        elif self.reduce_on_frequency(epoch + 1):
            self.reduce_lr(lr)
        else:
            self.best = current

        self.increment()

        print("\nLearning rate:", lr)
