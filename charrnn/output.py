from __future__ import print_function

import functools

import numpy as np

from . const import CHAR_IND

__all__ = 'print_model', 'printer', 'p'

p = functools.partial(print, sep='\t')

char_ind = np.array(list(CHAR_IND))


def print_y(y):
    print(''.join([char_ind[i][0] for i in y]))


def print_x(X, ind=0):
    print(''.join([char_ind[i][0] for i in X[ind]]))


def print_model(model, args):
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
    p('Train Batches:', len(t_train) // args.batch)
    p('Val Batches:', len(t_val) // (args.batch))
    p('Window Size:', args.window)
    p('Outout File:', args.model)
    p('Log Directory:', args.log_dir)
    print('\n', end='')
