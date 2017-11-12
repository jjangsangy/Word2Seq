from __future__ import print_function

import os
import h5py


def load_chars(model_path):
    with h5py.File(model_path, 'a') as f:
        return list(f['model_chars'].value.decode('utf8'))


def get_text(datasets):
    """
    Grab all the text dataset in the datasets directory
    """
    text = []
    for f in os.listdir(datasets):
        filepath = '/'.join([datasets, f])
        if f.startswith('.'):
            continue
        with open(filepath, encoding='utf8 ') as fp:
            try:
                text.append(fp.read())
            except UnicodeDecodeError:
                print('Could Not Read:', filepath)

    print('Total Files:', len(text), sep='\t')
    return '\n'.join(text)
