# -*- coding: utf-8 -*-
"""
Helper script that prints out the config information of a keras h5 file
"""
import json
import h5py
import argparse
import sys


def cli():
    parser = argparse.ArgumentParser(
        description='Print out training config information',
    )
    parser.add_argument('--model', '-m', default='model.h5', type=str,
                        help='Specify a model file')
    parser.add_argument('--config', '-c', default='training_config', type=str,
                        help='Specify the type of config to print out')
    return parser.parse_args()


def main():
    args = cli()
    f = h5py.File(args.model)
    config = json.loads(f.attrs[args.config].decode('utf8'))
    print(json.dumps(config, sort_keys=True, indent=4))
    f.close()


if __name__ == '__main__':
    sys.exit(main())
