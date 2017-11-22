# -*- coding: utf-8 -*-
"""
Decoder module for CharRNN
"""
from __future__ import print_function

import random
import sys
import h5py

import numpy as np

from . const import CHARS, IND_CHAR, CHAR_IND
from . text import random_text
from . utils import get_window

np.seterr(divide='ignore')

__all__ = 'sample', 'random_sentence'


def print_char(char):
    """
    Prints a character and flushes
    """
    sys.stdout.write(char)
    sys.stdout.flush()


def sample(preds, t=1.0):
    """
    Helper function to sample from a probability distribution
    """
    # Set float64 for due to numpy multinomial sampling issue
    # (https://github.com/numpy/numpy/issues/8317)
    preds = preds.astype('float64')
    preds = np.exp(np.log(preds) / t)
    preds /= preds.sum()
    return np.argmax(np.random.multinomial(n=1, pvals=preds.squeeze(), size=1))


def random_sentence(text, beam_size):
    """
    Grabs a random selection from a text.
    """
    rand_point = random.randint(0, len(text) - 1)
    correction = text[rand_point:].find('.') + 2
    start_index = rand_point + correction
    return text[start_index: start_index + beam_size]


def run(args):
    """
    Main entry point for outputting trained network
    """
    import keras
    window = get_window(args.model)
    model = keras.models.load_model(args.model)
    text = random_text(args.datasets)
    sentence = random_sentence(text, window)

    model.summary()
    print('Using seed:', sentence, sep='\n', end='\n\n')

    print_char(sentence)

    for _ in range(args.output):

        x = np.zeros((args.batch, window, len(CHARS)))
        for t, char in enumerate(sentence):
            x[0, t, CHAR_IND[char]] = 1.

        preds = model.predict_on_batch(x)
        next_index = sample(preds[0], t=args.temperature)
        next_char = IND_CHAR[next_index]

        sentence = sentence[1:] + next_char

        print_char(next_char)

    print_char('\n')
