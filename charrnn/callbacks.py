# -*- coding: utf-8 -*-
"""
Custom keras callbacks modules
"""
import h5py

from keras.callbacks import ModelCheckpoint

__all__ = 'CharRNNCheckpoint',


class CharRNNCheckpoint(ModelCheckpoint):

    def __init__(self, filepath, window,  *args, **kwargs):
        self.window = window
        super().__init__(filepath, *args, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs=logs)
        with h5py.File(self.filepath) as h5file:
            h5file.attrs['window'] = self.window
