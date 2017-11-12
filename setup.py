"""
CharRNN
=======
Character Recurrent Neural Network
"""
import os

from setuptools import setup


def load_version(*filepath):
    filepath = os.sep.join(filepath)
    namespace = {}
    with open(filepath) as f:
        exec(compile(f.read(), filepath, 'exec'), {}, namespace)
    return namespace

ver = load_version('charrnn', 'version.py')

setup(
    name='charrnn',
    author='Sang Han',
    version=ver['__version__'],
    description='Automatic Web Article Summarizer',
    long_description='\n'.join(
        [
            open('README.rst', 'rb').read().decode('utf-8'),
        ]
    ),
    license='Apache License 2.0',
    url='https://github.com/jjangsangy/Word2Seq',
    author_email='sanghan@protonmail.com',
    include_package_data=True,
    packages=['charrnn'],
    entry_points={
        'console_scripts': [
            'charrnn=charrnn.__main__:main'
        ],
    },
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Unix Shell',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Text Processing :: Linguistic',
        'Topic :: Utilities',
    ],

)
