# -*- coding: utf-8 -*-
"""
Command Line Argument Parsing
"""
from argparse import ArgumentParser

from . version import __version__ as version
from . version import __build__ as build

__all__ = 'command_line',


def command_line():
    """
    Parameterze training and prediction scripts for encoder and decoder character RNN's
    """
    parser = ArgumentParser(prog='charrnn', description='Train a neural network')
    parser.add_argument('--version', action='version',
                        version='%(prog)s {version} {build}'.format(version=version, build=build))
    parser.add_argument('--verbose', '-v', action='count', default=0,
                        help='Keras verbose output')

    # Options
    model, window, batch, datasets = 'model.h5', 40, 128, 'datasets'
    parser.add_argument('--model', '-m', default=model, type=str, metavar='file',
                        help='Specify the model hdf5 file to save to or load from: [default]: {model}'.format(model=model))
    parser.add_argument('--batch', '-b',  metavar='size', default=batch,
                        type=int, help='Specify the input batch size for LSTM layers: [default]: {batch}'.format(batch=batch))
    parser.add_argument('--datasets', '-t', metavar='directory', default=datasets, type=str,
                        help='Specify the directory where the datasets are located [default]: {datasets}'.format(datasets=datasets))

    # Subparser
    subparsers = parser.add_subparsers(help='Help train or produce output from your neural network')

    # Setup Defaults
    encoder = subparsers.add_parser('train', help='Train your character recurrent neural net')
    encoder.set_defaults(which='encode')

    decoder = subparsers.add_parser('decode', help='Output from previously trained network')
    decoder.set_defaults(which='decode')

    # Encoder
    dropout, layers, log_dir, lr = 0.2, 3, None, None
    epochs, optimizer, monitor, split = 100, 'nadam', 'val_loss', 0.15
    steps, decay = 5, 0.5

    encoder.add_argument('--resume', action='count',
                         help='Resume from saved model file rather than creating a new model at {model}'.format(model=model))
    encoder.add_argument('--window', '-w', default=window, type=int, metavar='length',
                         help='Specify the size of the window size to train on: [default]: {window}'.format(window=window))
    encoder.add_argument('--steps', '-u', metavar='size', default=steps, type=int,
                         help='Step size to conserve memory and speedup training [default]: {steps}'.format(steps=steps))
    encoder.add_argument('--log_dir', '-r', default=log_dir, type=str, metavar='directory',
                         help='Specify the output directory for tensorflow logs: [default]: {log_dir}'.format(log_dir=log_dir))
    encoder.add_argument('--split', '-p', default=split, type=float, metavar='size',
                         help='Specify the split between validation and training data [default]: {split}'.format(split=split))
    encoder.add_argument('--layers', '-l', default=layers, type=int, metavar='deep',
                         help='Specify the number of layers deep of LSTM nodes: [default]: {layers}'.format(layers=layers))
    encoder.add_argument('--dropout', '-d', default=dropout, type=float, metavar='amount',
                         help='Amount of LSTM dropout to apply between 0.0 - 1.0: [default]: {dropout}'.format(dropout=dropout))
    encoder.add_argument('--epochs', '-e', default=epochs, type=int, metavar='num',
                         help='Specify for however many epochs to train over [default]: {epochs}'.format(epochs=epochs))
    encoder.add_argument('--optimizer', '-o', default=optimizer, type=str, metavar='optimizer',
                         help='Specify optimizer used to train gradient descent: [default]: {optimizer}'.format(optimizer=optimizer))
    encoder.add_argument('--monitor', '-n', default=monitor, type=str, metavar='monitor',
                         help='Specify value to monitor for training/building model: [defaut]: {monitor}'.format(monitor=monitor))
    encoder.add_argument('--lr', '-a', default=lr, type=float, metavar='rate',
                         help='Set the learning rate for gradient descet optimizer: [default]: {lr}'.format(lr=lr))
    encoder.add_argument('--decay', '-y', default=decay, type=float, metavar='rate',
                         help='The rate in which to reduce the learning rate on plateau [default]: {decay}'.format(decay=decay))

    # Decoder
    layers, temperature, output = 3, 0.8, 4000

    decoder.add_argument('--temperature', '-t', default=temperature, type=float, metavar='t',
                         help='Set the temperature value for prediction on batch: [default]: {temperature}'.format(temperature=temperature))
    decoder.add_argument('--output', '-o', default=output, type=int, metavar='size',
                         help='Set the desired size of the characters decoded: [default]: {output}'.format(output=output))

    return parser.parse_args()
