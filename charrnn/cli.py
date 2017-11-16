# -*- coding: utf-8 -*-
from argparse import ArgumentParser


def command_line():
    """
    Parameterze training and prediction scripts for encoder and decoder character RNN's
    """
    model, datasets, window, batch = 'model.h5', 'datasets', 40, 128
    parser = ArgumentParser(prog='charrnn', description='Train a neural network')

    parser.add_argument('--verbose', '-v', action='count', default=0,
                        help='Keras verbose output')
    parser.add_argument('--model', '-m', metavar='file', default=model,
                        help=f'Specify the model hdf5 file to save to or load from: [default]: {model}')
    parser.add_argument('--window', '-w', default=window, type=int, metavar='length',
                        help=f'Specify the size of the window size to train on: [default]: {window}')
    parser.add_argument('--batch', '-b',  metavar='size', default=batch,
                        type=int, help=f'Specify the input batch size for LSTM layers: [default]: {batch}')
    parser.add_argument('--datasets', '-t', metavar='directory', default=datasets, type=str,
                        help=f'Specify the directory where the datasets are located [default]: {datasets}')

    # Subparser
    subparsers = parser.add_subparsers(help='Help train or produce output from your neural network')

    # Setup Defaults
    encoder = subparsers.add_parser('train', help='Train your character recurrent neural net')
    encoder.set_defaults(which='encode')

    decoder = subparsers.add_parser('decode', help='Output from previously trained network')
    decoder.set_defaults(which='decode')

    # Encoder
    dropout, layers, log_dir = 0.2, 3, None
    epochs, optimizer, monitor, split = 50, 'nadam', 'val_loss', 0.15

    encoder.add_argument('--log_dir', '-r', default=log_dir, metavar='directory',
                         help=f'Specify the output directory for tensorflow logs: [default]: {log_dir}')
    encoder.add_argument('--split', '-p', default=split, type=float, metavar='size',
                         help=f'Specify the split between validation and training data [default]: {split}')
    encoder.add_argument('--layers', '-l', default=3, type=int, metavar='deep',
                         help=f'Specify the number of layers deep of LSTM nodes: [default]: {layers}')
    encoder.add_argument('--dropout', '-d', default=dropout, type=float, metavar='amount',
                         help=f'Amount of LSTM dropout to apply between 0.0 - 1.0: [default]: {dropout}')
    encoder.add_argument('--resume', action='count',
                         help=f'Resume from saved model file rather than creating a new model at {model}')
    encoder.add_argument('--epochs', '-e', default=epochs, type=int, metavar='num',
                         help=f'Specify for however many epochs to train over [default]: {epochs}')
    encoder.add_argument('--optimizer', '-o', default=optimizer, type=str, metavar='optimizer',
                         help=f'Specify optimizer used to train gradient descent: [default]: {optimizer}')
    encoder.add_argument('--monitor', '-n', default=monitor, type=str, metavar='monitor',
                         help=f'Specify value to monitor for training/building model: [defaut]: {monitor}')

    # Decoder
    layers, temperature, output = 3, 0.8, 4000

    decoder.add_argument('--temperature', '-t', default=float(temperature), type=float, metavar='t',
                         help=f'Set the temperature value for prediction on batch: [default]: {temperature}')
    decoder.add_argument('--output', '-o', default=int(output), type=int, metavar='size',
                         help=f'Set the desired size of the characters decoded: [default]: {output}', )

    return parser.parse_args()
