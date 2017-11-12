# -*- coding: utf-8 -*-
import sys
import argparse

from .cli import command_line


def main():
    args = command_line()

    if args.which == 'encode':
        from .train import run

    if args.which == 'decode':
        from .decoder import run

    run(args)

if __name__ == '__main__':
    sys.exit(main())
