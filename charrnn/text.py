# -*- coding: utf-8 -*-
from __future__ import print_function

import os

from string import whitespace, punctuation, ascii_letters, digits

CHARS = sorted(whitespace + punctuation + ascii_letters + digits)
CHAR_IND = dict((c, i) for i, c in enumerate(CHARS))
IND_CHAR = dict((i, c) for i, c in enumerate(CHARS))


def get_text(directory):
    """
    Grab all the text dataset in the directory
    """
    text = []
    for dataset in os.listdir(directory):
        filepath = '/'.join([directory, dataset])
        if dataset.startswith('.'):
            continue
        with open(filepath, encoding='utf-8') as dset:
            try:
                text.append(dset.read())
            except UnicodeDecodeError:
                print('Could Not Read:', filepath)
    print('Total Files:', len(text), sep='\t')
    return '\n'.join(text)
