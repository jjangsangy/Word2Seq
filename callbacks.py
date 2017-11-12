import h5py

from keras.callbacks import ModelCheckpoint


class CharRNNCheckpoint(ModelCheckpoint):

    def __init__(self, chars, *args, **kwargs):
        self.chars = chars
        super().__init__(*args, **kwargs)

    def on_epoch_end(self, *args, **kwargs):
        super().on_epoch_end(*args, **kwargs)
        with h5py.File(self.filepath, 'a') as f:
            f['model_chars'] = ''.join(self.chars).encode('utf8')
