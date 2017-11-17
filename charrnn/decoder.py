# -*- coding: utf-8 -*-
"""
Decoder module for CharRNN
"""
from __future__ import print_function

import keras
import sys
import random
import os

import numpy as np

from . cli import command_line
from . text import CHARS, IND_CHAR, CHAR_IND

np.seterr(divide='ignore')


def random_text(directory):
    """
    Reads a random file inside a directory of text files
    """
    datasets = [i for i in os.listdir(directory) if not i.startswith('.')]
    filepath = '/'.join([directory, random.choice(datasets)])
    with open(filepath, 'rt', encoding='utf-8') as dset:
        return dset.read()


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
    rand_point = random.randint(0, len(text) - 1)
    correction = text[rand_point:].find('.') + 2
    start_index = rand_point + correction
    return text[start_index: start_index + beam_size]


def run(args):
    """
    Main entry point for outputting trained network
    """
    text = random_text(args.datasets)
    window_size = args.window

    model = keras.models.load_model(args.model)
    print(model.summary())

    sentence = random_sentence(text, window_size)

    generated = sentence

    print('Using seed:', generated, sep='\n', end='\n\n')

    sys.stdout.write(generated)
    sys.stdout.flush()

    for _ in range(args.output):

        x = np.zeros((args.batch, window_size, len(CHARS)))
        for t, char in enumerate(sentence):
            x[0, t, CHAR_IND[char]] = 1.

        preds = model.predict_on_batch(x)
        next_index = sample(preds[0], t=args.temperature)
        next_char = IND_CHAR[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

        sys.stdout.write(next_char)
        sys.stdout.flush()

    sys.stdout.write('\n')
