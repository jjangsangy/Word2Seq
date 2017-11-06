# -*- coding: utf-8 -*-
"""
Training a Recurrent Neural Network for Text Generation
=======================================================
This implements a char-rnn, which was heavily inspired from
Andrei Karpathy's work on text generation and adapted from
example code introduced by keras.
(http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

It is recommended to run this script on GPU, as
recurrent networks are quite computationally intensive.
Make sure your corpus has >100k characters at least, but
for the best >1M characters. That is around the size,
of Harry Potter Book 7.
"""
from __future__ import print_function

import keras
import operator
import pathlib
import random
import sys
import os
import argparse
import functools

import numpy as np

from builtins import str as stringify
from itertools import chain
from keras.callbacks import ModelCheckpoint, Callback, TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras.layers.core import Dense, Activation, Dropout, Masking
from keras.engine.topology import Input
from keras.layers.recurrent import LSTM, RNN
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.optimizers import RMSprop
from past.builtins import basestring
from keras.models import load_model
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Flatten


def build_model(args):
    """
    Build a Stateful Stacked LSTM Network with n-stacks specified by args.layers
    """
    layers = list(reversed(range(1, args.layers)))
    params = dict(return_sequences=True, dropout=args.dropout, stateful=True,
                  batch_input_shape=(args.batch, args.window, len(args.chars)))
    model = Sequential()

    while layers:
        layers.pop()
        model.add(LSTM(args.batch, **params))
    else:
        # Last Layer is Flat
        del params['return_sequences']
        model.add(LSTM(args.batch, **params))

    model.add(Dense(len(args.chars), activation='softmax', name='output'))

    model.compile(loss=categorical_crossentropy,
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    return model


def command_line(setup='encoder'):
    """
    Parameterze training and prediction scripts for encoder and decoder character rnn's
    """
    parser = argparse.ArgumentParser(description='Train a neural network')

    parser.add_argument('--verbose', '-v', action='count', default=0,
                        help='Keras verbose output')
    parser.add_argument('--batch', '-b',  metavar='size', default=128,
                        type=int, help='Specify the input batch size')
    parser.add_argument('--model', '-m', metavar='file',
                        default='models/model.h5',
                        help='Specify the output model hdf5 file to save to: [default]: models/model.h5')
    parser.add_argument('--layers', '-l', default=3, type=int, metavar='deep',
                        help='Specify the number of layers deep of LSTM nodes: [default]: 3')

    parser.add_argument('--dropout', '-d', default=0.2, type=float, metavar='amount',
                        help='Amount of LSTM dropout to apply between 0.0 - 1.0: [default]: 0.2')
    parser.add_argument('--resume', action='count',
                        help='Resume from saved model file rather than creating a new model')
    parser.add_argument('--window', '-w', default=40, type=int, metavar='length',
                        help='Specify the size of the window size to train on: [default]: 40')
    parser.add_argument('--log_dir', '-r', default=None, metavar='directory',
                        help='Specify the output directory for tensorflow logs: [default]: None')
    parser.add_argument('--split', '-p', default=0.15, type=float, metavar='size',
                        help='Specify the split between validation and training data [default]: 0.15')

    if setup == 'decoder':
        parser.add_argument('--temperature', '-t', default=1.0, type=float, metavar='t',
                            help='Set the temperature value for prediction on batch: [default]: 1.0')
        parser.add_argument('--output', '-s', default=2000, type=int, metavar='size',
                            help='Set the desired size of the characters decoded: [default]: 20000', )
    if setup == 'encoder':
        parser.add_argument('--epochs', '-e', default=50, type=int, metavar='num',
                            help='Specify for however many epochs to train over [default]: 50')

    args = parser.parse_args()
    args.sentences = []
    args.next_chars = []
    return args


def printer(args):
    """
    Helper print function on statistics
    """
    p = functools.partial(print, sep='\t')
    p('Total Chars:', len(args.chars))
    p('Corpus Length:', len(args.text))
    p('NB Sequences:', len(args.sentences),
      'of [{window}]'.format(window=args.window))
    p('Outout File:', args.model)
    p('Log Directory:', args.log_dir)
    p('LSTM Layers:', args.layers)
    p('LSTM Dropout:', args.dropout)


def get_text(datasets):
    """
    Grab all the text dataset in the datasets directory
    """
    text = []
    for f in os.listdir(datasets):
        filepath = '/'.join([datasets, f])
        if f.startswith('.'):
            continue
        with open(filepath, encoding='utf8 ') as fp:
            try:
                text.append(fp.read())
                print('Read:', filepath, sep='\t')
            except UnicodeDecodeError:
                print('Could Not Read', filepath, sep='\t')

    return '\n'.join(text)


def train_validation_split(args):
    """
    Split training and validation data specified by args.split
    """
    v_split = round((len(args.X) // args.batch) * (1 - args.split)) * args.batch
    args.x_train, args.y_train = args.X[:v_split], args.y[:v_split]
    args.x_val, args.y_val = args.X[v_split:], args.y[v_split:]
    return args


def parameterize(args):
    """
    Parameterize argparse namespace with more parameters generated from dataset
    """
    args.text = get_text('datasets')
    args.chars = sorted(list(set(args.text)))
    args.char_indices = dict((c, i) for i, c in enumerate(args.chars))
    args.indices_char = dict((i, c) for i, c in enumerate(args.chars))

    for i in range(0, len(args.text) - args.window, 1):
        args.sentences.append(args.text[i: i + args.window])
        args.next_chars.append(args.text[i + args.window])

    # Print all the params
    printer(args)

    max_window = len(args.sentences) - (len(args.sentences) % args.batch)
    args.sentences = args.sentences[0: max_window]
    args.next_chars = args.next_chars[0: max_window]

    args.X = np.zeros((max_window, args.window, len(args.chars)), dtype=np.bool)
    args.y = np.zeros((max_window,              len(args.chars)), dtype=np.bool)

    for i, sentence in enumerate(args.sentences):
        for t, char in enumerate(sentence):
            args.X[i, t, args.char_indices[char]] = 1
        args.y[i, args.char_indices[args.next_chars[i]]] = 1

    return train_validation_split(args)


def main():
    """
    Main entry point for training network
    """
    args = parameterize(command_line())

    # Build Model
    model = (
        load_model(args.model) if os.path.exists(args.model) and args.resume else build_model(args)
    )

    print(model.summary())

    callbacks = [
        ModelCheckpoint(args.model, save_best_only=True, monitor='val_loss', verbose=args.verbose),
        ReduceLROnPlateau(factor=0.2, patience=2, monitor='val_loss', verbose=args.verbose)
    ]

    if args.log_dir:
        callbacks.append(TensorBoard(log_dir=args.log_dir, histogram_freq=10,
                                     write_grads=True, batch_size=args.batch))

    # Go Get Some Coffee
    model.fit(x=args.x_train, y=args.y_train,
              batch_size=args.batch,
              epochs=args.epochs,
              callbacks=callbacks,
              shuffle=False,
              validation_data=(args.x_val, args.y_val))


if __name__ == '__main__':
    sys.exit(main())
