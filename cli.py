from argparse import ArgumentParser, Namespace


def command_line(setup):
    """
    Parameterze training and prediction scripts for encoder and decoder character RNN's
    """
    model, layers, batch_size = 'models/model.h5', 3, 128
    dropout, window, log_dir, split = 0.2, 40, None, 0.15

    parser = ArgumentParser(description='Train a neural network')

    parser.add_argument('--verbose', '-v', action='count', default=0,
                        help='Keras verbose output')
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
        temperature, output = 0.8, 4000
        parser.add_argument('--temperature', '-t', default=float(temperature), type=float, metavar='t',
                            help=f'Set the temperature value for prediction on batch: [default]: {temperature}')
        parser.add_argument('--output', '-o', default=int(output), type=int, metavar='size',
                            help=f'Set the desired size of the characters decoded: [default]: {output}', )

    if setup == 'encoder':
        epochs, optimizer, monitor = 50, 'nadam', 'val_loss'
        parser.add_argument('--resume', action='count',
                            help=f'Resume from saved model file rather than creating a new model at {model}')
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
