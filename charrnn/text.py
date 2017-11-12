# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import h5py

from string import whitespace, punctuation, ascii_letters, digits

CHARS = sorted(whitespace + punctuation + ascii_letters + digits)
CHAR_IND = dict((c, i) for i, c in enumerate(CHARS))
IND_CHAR = dict((i, c) for i, c in enumerate(CHARS))


def get_text(datasets):
    """
    Grab all the text dataset in the datasets directory
    """
    text = []
    for f in os.listdir(datasets):
        filepath = '/'.join([datasets, f])
        if f.startswith('.'):
            continue
        with open(filepath, encoding='utf-8') as fp:
            try:
                text.append(fp.read())
            except UnicodeDecodeError:
                print('Could Not Read:', filepath)

    print('Total Files:', len(text), sep='\t')

    return '\n'.join(text)
