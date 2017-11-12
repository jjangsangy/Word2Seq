# -*- coding: utf-8 -*-
"""
Module for training CharRNN
"""
from __future__ import print_function

import os
import functools

import keras

import numpy as np

from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras.losses import categorical_crossentropy
from keras.models import Sequential, load_model

from .cli import command_line
from .text import get_text
from .text import CHARS, CHAR_IND, IND_CHAR


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
                  batch_input_shape=(args.batch, args.window, len(CHARS)))
    model = Sequential()

    while layers:
        layers.pop()
        model.add(LSTM(args.batch, **params))
    else:
        # Last Layer is Flat
        del params['return_sequences']
        model.add(LSTM(args.batch, **params))

    model.add(Dense(len(CHARS), name='softmax', activation='softmax'))

    model.compile(loss=categorical_crossentropy,
                  optimizer=args.optimizer,
                  metrics=['accuracy'])

    print_model(model, args)

    return model


def printer(args):
    """
    Helper print function on statistics
    """
    p('Corpus Length:', len(args.text))
    p('NB Sequences:', len(args.sentences),
      'of [{window}]'.format(window=args.window))
    p('Outout File:', args.model)
    p('Log Directory:', args.log_dir)


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
    args.text = get_text(args.datasets)
    args.sentences = []
    args.next_chars = []

    for i in range(0, len(args.text) - args.window, 1):
        args.sentences.append(args.text[i: i + args.window])
        args.next_chars.append(args.text[i + args.window])

    # Print all the params
    printer(args)

    max_window = len(args.sentences) - (len(args.sentences) % args.batch)
    args.sentences = args.sentences[0: max_window]
    args.next_chars = args.next_chars[0: max_window]

    args.X = np.zeros((max_window, args.window, len(CHARS)), dtype=np.bool)
    args.y = np.zeros((max_window,              len(CHARS)), dtype=np.bool)

    for i, sentence in enumerate(args.sentences):
        for t, char in enumerate(sentence):
            args.X[i, t, CHAR_IND[char]] = 1
        args.y[i, CHAR_IND[args.next_chars[i]]] = 1

    return train_validation_split(args)


def run(args):
    """
    Main entry point for training network
    """
    args = parameterize(args)
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
