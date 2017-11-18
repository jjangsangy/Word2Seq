# -*- coding: utf-8 -*-
"""
Module for training CharRNN
"""
from __future__ import print_function

import os
import functools
import operator


import keras
import numpy as np

from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras.losses import categorical_crossentropy
from keras.models import Sequential, load_model

from chainmap import ChainMap

from . text import get_text
from . const import CHARS, CHAR_IND, IND_CHAR

__all__ = 'print_model', 'printer', 'parameterize', 'gen', 'get_optimzer', 'build_model'


p = functools.partial(print, sep='\t')


def print_model(model, args):
    p('LSTM Layers:', args.layers)
    p('LSTM Dropout:', args.dropout)
    p('Optimizer:', args.optimizer)
    p('Optim Config:', args.optimizer_config)
    p('Learning Rate:', args.lr)
    p('Decay Rate:', args.decay)
    model.summary()
    print('\n', end='')


def printer(args):
    """
    Helper print function on statistics
    """
    p('Corpus Length:', len(args.text))
    p('NB Sequences:', len(args.sentences),
      'of [{window}]'.format(window=args.window))
    p('Outout File:', args.model)
    p('Log Directory:', args.log_dir)
    p('Step Size:', args.steps)
    p('Val Split:', args.split)
    print('\n', end='')


def parameterize(args):
    """
    Parameterize argparse namespace with more parameters generated from dataset
    """
    args.text = get_text(args.datasets)
    args.sentences = []
    args.next_chars = []

    for i in range(0, len(args.text) - args.window, args.steps):
        args.sentences.append(args.text[i: i + args.window])
        args.next_chars.append(args.text[i + args.window])

    # Print all the params
    printer(args)

    max_window = len(args.sentences) - (len(args.sentences) % args.batch)
    args.sentences = args.sentences[0: max_window]
    args.next_chars = args.next_chars[0: max_window]

    X = np.zeros((max_window, args.window, len(CHARS)), dtype=np.bool)
    y = np.zeros((max_window,              len(CHARS)), dtype=np.bool)

    for i, sentence in enumerate(args.sentences):
        for t, char in enumerate(sentence):
            X[i, t, CHAR_IND[char]] = 1
        y[i, CHAR_IND[args.next_chars[i]]] = 1

    return X, y


def gen(X, y, batch_size):
    re_x = X.reshape((X.shape[0] // batch_size, batch_size, X.shape[1], X.shape[2]))
    re_y = y.reshape((y.shape[0] // batch_size, batch_size, y.shape[1]))
    while True:
        for i in range(len(re_x)):
            yield re_x[i], re_y[i]


def get_optimzer(opt, **kwargs):
    grab = operator.attrgetter(opt)
    optimizer = grab(keras.optimizers)
    return optimizer(**kwargs)


def build_model(args):
    """
    Build a Stateful Stacked LSTM Network with n-stacks specified by args.layers
    """
    layers = list(reversed(range(1, args.layers)))
    params = dict(return_sequences=True, stateful=True, dropout=args.dropout,
                  batch_input_shape=(args.batch, args.window, len(CHARS)))
    opt_args = ChainMap({'lr': args.lr},
                        dict([i.split('=') for i in args.optimizer_config.split()]))
    optimizer = get_optimzer(args.optimizer, **dict(opt_args))
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
                  optimizer=optimizer,
                  metrics=['accuracy'])

    print_model(model, args)

    return model


def run(args):
    """
    Main entry point for training network
    """
    # Build Model
    model = (
        load_model(args.model) if os.path.exists(args.model) and args.resume else build_model(args)
    )

    callbacks = [
        ModelCheckpoint(args.model, save_best_only=True,
                        monitor=args.monitor, verbose=args.verbose),
        ReduceLROnPlateau(factor=args.decay, patience=0,
                          monitor=args.monitor, verbose=args.verbose),
    ]

    if args.log_dir:
        callbacks.append(TensorBoard(log_dir=args.log_dir, histogram_freq=10,
                                     write_grads=True, batch_size=args.batch))

    X, y = parameterize(args)

    v_split = round((len(X) // args.batch) * (1 - args.split)) * args.batch

    x_train, y_train = X[:v_split], y[:v_split]
    x_val, y_val = X[v_split:], y[v_split:]

    # Go Get Some Coffee
    model.fit_generator(generator=gen(x_train, y_train, args.batch),
                        steps_per_epoch=len(x_train) // args.batch,
                        validation_data=gen(x_val, y_val, args.batch),
                        validation_steps=len(x_val) // args.batch,
                        epochs=args.epochs,
                        callbacks=callbacks,
                        use_multiprocessing=True,
                        shuffle=False)
