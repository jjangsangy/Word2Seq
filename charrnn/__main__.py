# -*- coding: utf-8 -*-
"""
Main Entrypoint for CharRNN

Allows for running module with python3 -m
"""

import sys

from . cli import command_line

__all__ = 'main'


def main():
    """
    Main Entrypoint for CLI
    """
    args = command_line()

    # Do Not Import Keras Until Needed
    if args.which == 'encode':
        from .train import run
    if args.which == 'decode':
        from .decoder import run

    run(args)

if __name__ == '__main__':
    sys.exit(main())
