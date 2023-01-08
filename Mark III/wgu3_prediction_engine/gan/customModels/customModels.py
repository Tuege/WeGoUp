import tensorflow as tf
from tensorflow import keras
from keras import layers


class DiscriminatorModel (keras.Sequential):
    def __init__(self, n_layers=None, name='discriminator'):
        if n_layers is None:
            self.n_layers = [
                keras.Input(shape=(28, 28, 1)),
                layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.GlobalMaxPooling2D(),
                layers.Dense(1)
            ]
        super().__init__(self, n_layers, name)


class GeneratorModel (keras.Sequential):
    def __init__(self, n_layers=None, name='generator', latent_dim=128):
        if n_layers is None:
            self.n_layers = [
                keras.Input(shape=(latent_dim,)),
                # We want to generate 128 coefficients to reshape into a 7x7x128 map
                layers.Dense(7 * 7 * 128),
                layers.LeakyReLU(alpha=0.2),
                layers.Reshape((7, 7, 128)),
                layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid")
            ]
        super().__init__(self, n_layers, name)