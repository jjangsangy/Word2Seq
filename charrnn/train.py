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

__all__ = 'print_model', 'printer', 'gen', 'gen_text', 'get_optimzer', 'build_model'


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


def gen_text(text, args):
    last_window = len(text) - args.window
    for i in range(0, last_window):
        yield text[i: i + args.window], text[i + args.window]


def gen(text, args):
    g = gen_text(text, args)
    while True:
        try:
            train_window, test_window = [], []
            for i in range(args.batch):
                train, test = next(g)
                X = np.zeros((args.batch, args.window, len(CHARS)), dtype=np.bool)
                y = np.zeros((args.batch,              len(CHARS)), dtype=np.bool)
                train_window.append(train)
                test_window.append(test)
            for j, sentence in enumerate(train_window):
                for t, char in enumerate(sentence):
                    X[j, t, CHAR_IND[char]] = True
                y[j, CHAR_IND[test_window[j]]] = True
            yield X, y
        except StopIteration:
            g = gen_text(text, args)


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
    printer(args)

    text = get_text(args.datasets)

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

    v_split = round((len(text) // args.batch) * (1 - args.split)) * args.batch

    t_train, t_val = text[:v_split], text[v_split:]

    # Go Get Some Coffee
    model.fit_generator(generator=gen(t_train, args),
                        steps_per_epoch=len(t_train) // args.batch,
                        validation_data=gen(t_val, args),
                        validation_steps=len(t_val) // args.batch,
                        epochs=args.epochs,
                        callbacks=callbacks,
                        use_multiprocessing=True,
                        shuffle=False)
