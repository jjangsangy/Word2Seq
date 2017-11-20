# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import random

import numpy as np


from . const import TRANS_TABLE

__all__ = 'get_text', 'random_text', 'translate'


def translate(text, batch=128):
    trans = text.translate(TRANS_TABLE)
    return np.fromstring(trans[:len(trans) - len(trans) % batch], dtype='b')


def random_text(directory):
    """
    Reads a random file inside a directory of text files
    """
    datasets = [i for i in os.listdir(directory) if not i.startswith('.')]
    filepath = '/'.join([directory, random.choice(datasets)])
    with open(filepath, 'rt', encoding='utf-8') as dset:
        return dset.read()


def get_text(directory):
    """
    Grab all the text dataset in the directory
    """
    text = []
    for dataset in os.listdir(directory):
        filepath = '/'.join([directory, dataset])
        if dataset.startswith('.'):
            continue
        with open(filepath, encoding='utf-8') as dset:
            try:
                text.append(dset.read())
            except UnicodeDecodeError:
                print('Could Not Read:', filepath)
    print('Total Files:', len(text), sep='\t')
    return '\n'.join(text)
