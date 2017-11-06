# -*- coding: utf-8 -*-
"""
Decoder module for Char RNN
"""
from __future__ import print_function

import keras
import sys
import random

import numpy as np

from train import parameterize, command_line

args = parameterize(command_line('decoder'))


model_output = args.model
chars = args.chars
text = args.text
window_size = args.window
indices_char = args.indices_char
char_indices = args.char_indices


np.seterr(divide='ignore')


def sample(preds, t=1.0):
    """Helper function to sample from a probability distribution
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


model = keras.models.load_model(model_output)
print(model.summary())

sentence = random_sentence(text, window_size)


generated = sentence


print('Using seed:', generated, sep='\n', end='\n\n')

sys.stdout.write(generated)
sys.stdout.flush()

for _ in range(args.output):

    x = np.zeros((args.batch, window_size, len(chars)))
    for t, char in enumerate(sentence):
        x[0, t, char_indices[char]] = 1.

    preds = model.predict_on_batch(x)
    next_index = sample(preds[0], t=args.temperature)
    next_char = indices_char[next_index]

    generated += next_char
    sentence = sentence[1:] + next_char

    sys.stdout.write(next_char)
    sys.stdout.flush()

sys.stdout.write('\n')
