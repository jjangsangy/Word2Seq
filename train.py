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
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.optimizers import RMSprop
from past.builtins import basestring
from keras.models import load_model
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Flatten

from keras.layers.recurrent import LSTM

p = functools.partial(print, sep='\t')


def print_model(model, args):
    p('LSTM Layers:', args.layers)
    p('LSTM Dropout:', args.dropout)
    p('LSTM Optimizer:', args.optimizer)
    print(model.summary())


def build_model(args):
    """
    Build a Stateful Stacked LSTM Network with n-stacks specified by args.layers
    """
    layers = list(reversed(range(1, args.layers)))
    params = dict(return_sequences=True, stateful=True, dropout=args.dropout,
                  batch_input_shape=(args.batch, args.window, len(args.chars)))
    model = Sequential()

    while layers:
        layers.pop()
        model.add(LSTM(args.batch, **params))
    else:
        # Last Layer is Flat
        del params['return_sequences']
        model.add(LSTM(args.batch, **params))

    model.add(Dense(len(args.chars), name='softmax', activation='softmax'))

    model.compile(loss=categorical_crossentropy,
                  optimizer=args.optimizer,
                  metrics=['accuracy'])

    print_model(model, args)

    return model


def command_line(setup='encoder'):
    """
    Parameterze training and prediction scripts for encoder and decoder character rnn's
    """
    model, layers, batch_size, dropout, window, log_dir, split = 'models/model.h5', 3, 128, 0.2, 40, None, 0.15
    parser = argparse.ArgumentParser(description='Train a neural network')

    parser.add_argument('--verbose', '-v', action='count', default=0,
                        help='Keras verbose output')
    parser.add_argument('--resume', action='count',
                        help=f'Resume from saved model file rather than creating a new model at {model}')
    parser.add_argument('--batch', '-b',  metavar='size', default=batch_size,
                        type=int, help=f'Specify the input batch size for LSTM lyaers: [default]: {batch_size}')
    parser.add_argument('--model', '-m', metavar='file', default=model,
                        help=f'Specify the output model hdf5 file to save to: [default]: {model}')
    parser.add_argument('--layers', '-l', default=3, type=int, metavar='deep',
                        help=f'Specify the number of layers deep of LSTM nodes: [default]: {layers}')

    parser.add_argument('--dropout', '-d', default=dropout, type=float, metavar='amount',
                        help=f'Amount of LSTM dropout to apply between 0.0 - 1.0: [default]: {dropout}')
    parser.add_argument('--window', '-w', default=window, type=int, metavar='length',
                        help=f'Specify the size of the window size to train on: [default]: {window}')
    parser.add_argument('--log_dir', '-r', default=log_dir, metavar='directory',
                        help=f'Specify the output directory for tensorflow logs: [default]: {log_dir}')
    parser.add_argument('--split', '-p', default=split, type=float, metavar='size',
                        help=f'Specify the split between validation and training data [default]: {split}')

    if setup == 'decoder':
        temperature, output = 0.8, 2000
        parser.add_argument('--temperature', '-t', default=float(temperature), type=float, metavar='t',
                            help=f'Set the temperature value for prediction on batch: [default]: ${temperature}')
        parser.add_argument('--output', '-o', default=int(output), type=int, metavar='size',
                            help=f'Set the desired size of the characters decoded: [default]: ${output}', )

    if setup == 'encoder':
        epochs, optimizer, monitor = 50, 'nadam', 'val_loss'
        parser.add_argument('--epochs', '-e', default=epochs, type=int, metavar='num',
                            help=f'Specify for however many epochs to train over [default]: {epochs}')
        parser.add_argument('--optimizer', '-o', default=optimizer, type=str, metavar='optimizer',
                            help=f'Specify optimizer used to train gradient descent: [default]: {optimizer}')
        parser.add_argument('--monitor', '-n', default=monitor, type=str, metavar='monitor',
                            help=f'Specify value to monitor for training/building model: [defaut]: {monitor}')

    args = parser.parse_args()
    args.sentences = []
    args.next_chars = []
    return args


def printer(args):
    """
    Helper print function on statistics
    """
    p('Total Chars:', len(args.chars))
    p('Corpus Length:', len(args.text))
    p('NB Sequences:', len(args.sentences),
      'of [{window}]'.format(window=args.window))
    p('Outout File:', args.model)
    p('Log Directory:', args.log_dir)


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
                p('Reading File:', filepath)
            except UnicodeDecodeError:
                p('Could Not Read:', filepath)
    p('Total Files:', len(text), '\n')
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
    args = parameterize(command_line('encoder'))

    # Build Model
    model = (
        load_model(args.model) if os.path.exists(args.model) and args.resume else build_model(args)
    )

    callbacks = [
        ModelCheckpoint(args.model, save_best_only=True,
                        monitor=args.monitor, verbose=args.verbose),
        ReduceLROnPlateau(factor=0.2, patience=2,
                          monitor=args.monitor, verbose=args.verbose),
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
