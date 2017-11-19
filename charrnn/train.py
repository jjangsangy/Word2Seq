# -*- coding: utf-8 -*-
"""
Module for training CharRNN
"""
from __future__ import print_function

import os
import functools
import operator


import numpy as np

from chainmap import ChainMap

from . text import get_text
from . const import CHARS, CHAR_IND, IND_CHAR

__all__ = 'print_model', 'printer', 'gen_batch', 'get_optimzer', 'build_model', 'tweak_lr'


p = functools.partial(print, sep='\t')


def print_model(model, args):
    p('Step Size:', args.steps)
    p('Val Split:', args.split)
    p('LSTM Layers:', args.layers)
    p('LSTM Dropout:', args.dropout)
    p('Optimizer:', args.optimizer)
    p('Learning Rate:', args.lr)
    p('Decay Rate:', args.decay)
    model.summary()
    print('\n', end='')


def printer(t_train, t_val, args):
    """
    Helper print function on statistics
    """
    p('Corpus Length:', len(t_train) + len(t_val))
    p('Train Batches:', len(t_train) // (args.batch * args.steps))
    p('Val Batches:', len(t_val) // (args.batch))
    p('Window Size:', args.window)
    p('Outout File:', args.model)
    p('Log Directory:', args.log_dir)
    print('\n', end='')


def tweak_lr(optimizer):
    default_values = {
        'nadam': 0.004,
        'adam': 0.001,
        'rmsprop': 0.001,
        'sgd': 0.01
    }
    return default_values[optimizer.lower()]


def get_optimzer(opt, **kwargs):
    import keras
    grab = operator.attrgetter(opt)
    optimizer = grab(keras.optimizers)
    return optimizer(**kwargs)


def gen_text(text, args):
    last_window = len(text) - args.window
    for i in range(last_window):
        yield text[i: i + args.window], text[i + args.window]


def gen_batch(text, args):
    """
    Infinitely generate batches of data of size args.batch
    """
    generator = gen_text(text, args)
    while True:
        try:
            # Create New Batch
            train_window, test_window = [], []
            X = np.zeros((args.batch, args.window, len(CHARS)), dtype=np.bool)
            y = np.zeros((args.batch,              len(CHARS)), dtype=np.bool)

            for _ in range(args.batch):
                train, test = next(generator)
                train_window.append(train)
                test_window.append(test)

            for i, sentence in enumerate(train_window):
                for t, char in enumerate(sentence):
                    X[i, t, CHAR_IND[char]] = True
                y[i, CHAR_IND[test_window[i]]] = True
            yield X, y

        except StopIteration:
            generator = gen_text(text, args)


def build_model(args):
    """
    Build a Stateful Stacked LSTM Network with n-stacks specified by args.layers
    """
    from keras.layers.recurrent import LSTM
    from keras.layers.core import Dense
    from keras.models import Sequential

    layers = list(reversed(range(1, args.layers)))
    params = dict(return_sequences=True, stateful=True, dropout=args.dropout,
                  batch_input_shape=(args.batch, args.window, len(CHARS)))

    opt_args = {
        'lr': args.lr if args.lr else tweak_lr(args.optimizer),
        'clipvalue': 4.0
    }
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

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    print_model(model, args)

    return model


def train_val_split(text, args):
    v_split = round((len(text) // args.batch) * (1 - args.split)) * args.batch
    return text[:v_split], text[v_split:]


def get_callbacks(args):
    from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau

    callbacks = [
        ModelCheckpoint(args.model, save_best_only=True,
                        monitor=args.monitor, verbose=args.verbose),
        ReduceLROnPlateau(factor=args.decay, patience=0, cooldown=args.steps,
                          monitor=args.monitor, verbose=args.verbose),
    ]
    if args.log_dir:
        callbacks.append(TensorBoard(log_dir=args.log_dir, histogram_freq=10,
                                     write_grads=True, batch_size=args.batch))
    return callbacks


def run(args):
    """
    Main entry point for training network
    """
    from keras.models import load_model
    # Build Model

    t_train, t_val = train_val_split(get_text(args.datasets), args)

    printer(t_train, t_val, args)

    model = (
        load_model(args.model) if os.path.exists(args.model) and args.resume else build_model(args)
    )

    # Go Get Some Coffee
    model.fit_generator(generator=gen_batch(t_train, args),
                        steps_per_epoch=len(t_train) // (args.batch * args.steps),
                        validation_data=gen_batch(t_val, args),
                        validation_steps=len(t_val) // args.batch,
                        epochs=args.epochs,
                        callbacks=get_callbacks(args),
                        use_multiprocessing=True,
                        shuffle=False)
