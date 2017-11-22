# -*- coding: utf-8 -*-
"""
Utility function for working with reading and writing to model files
"""

import h5py


def get_window(model):
    with h5py.File(model, 'r') as h5file:
        return h5file.attrs['window']
