# -*- coding: utf-8 -*-
"""
String constants used to one hot encode LSTM
"""

from string import whitespace, punctuation, ascii_letters, digits

CHARS = sorted(whitespace + punctuation + ascii_letters + digits)
CHAR_IND = dict((c, i) for i, c in enumerate(CHARS))
IND_CHAR = dict((i, c) for i, c in enumerate(CHARS))

TRANS_TABLE = dict((ord(k), v) for k, v in CHAR_IND.items())
