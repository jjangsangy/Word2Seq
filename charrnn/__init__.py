# -*- coding: utf-8 -*-
"""
Training a Recurrent Neural Network for Text Generation
=======================================================
This implements a char-rnn, which was heavily inspired from
Andrei Karpathy's work on text generation and adapted from
example code introduced by keras.
(http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

It is recommended to run this script on GPU, as
recurrent networks are quite computationally intensive.
Make sure your corpus has >100k characters at least, but
for the best >1M characters. That is around the size,
of Harry Potter Book 7.
"""

__author__ = 'Sang Han'
__title__ = 'charrnn'
__license__ = 'Apache Software License Version 2.0'

from . version import __version__
from . version import __build__
