# -*- coding: utf-8 -*-
"""
Custom keras callbacks modules
"""
import h5py
import numpy as np

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
        super(ModelCheckpoint, self).__init__(filepath, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs=logs)
        with h5py.File(self.filepath) as h5file:
            h5file.attrs['window'] = self.window


class AdvancedLRScheduler(Callback):
    '''
    Schedule learning rate when a monitored quantity does not
    improve over a period of time.
    '''

    def __init__(self, monitor='val_loss', patience=0,
                 verbose=False, mode='auto', decay_ratio=0.5):
        """
        monitor: quantity to be monitored.
        patience: number of epochs with no improvement
            after which lr will be lowered.
        verbose: verbosity mode.
        mode: one of {auto, min, max}. In 'min' mode,
            training will stop when the quantity
            monitored has stopped decreasing; in 'max'
            mode it will stop when the quantity
            monitored has stopped increasing.
        """
        super(Callback, self).__init__()

        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.wait = 0
        self.decay_ratio = decay_ratio

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        current = logs.get(self.monitor)
        current_lr = K.get_value(self.model.optimizer.lr)

        print("\nLearning rate:", current_lr)

        if self.monitor_op(current, self.best):
            self.best = current
            self.wait = 0
        else:
            if self.wait >= self.patience:
                if self.verbose > 0:
                    print('Epoch %05d: reducing learning rate' % (epoch))
                    current_lr = K.get_value(self.model.optimizer.lr)
                    new_lr = current_lr * self.decay_ratio
                    K.set_value(self.model.optimizer.lr, new_lr)
                    self.wait = 0

            self.wait += 1
