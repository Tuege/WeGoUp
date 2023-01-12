import tensorflow as tf
from tensorflow import keras
from keras import layers
import time
import numpy as np


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
        super().__init__(n_layers, name)


class GeneratorModel(keras.Sequential):  # , keras.callbacks.Callback):
    def __init__(self, shape=None, **kwargs):
        super().__init__(name='generator', **kwargs)
        self.add(layers.LSTM(100, return_sequences=True, input_shape=shape))
        self.add(layers.LSTM(100, return_sequences=False))
        self.add(layers.Dense(25))
        self.add(layers.Dense(1))

        callbacks = []
        self.callbacks = keras.callbacks.CallbackList(
            callbacks,
            # add_history=True,
            # add_progbar=verbose != 0,
            model=self,
            verbose=True,
            epochs=5,
            # steps=data_handler.inferred_steps,
        )

        # self.queue = tf.queue.FIFOQueue()

    @tf.function
    def train_step(self, inputs, labels):
        with tf.GradientTape() as tape:
            predictions = self(inputs, training=True)
            loss_value = self.compiled_loss(labels, predictions)
        grads = tape.gradient(loss_value, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        # train_acc_metric.update_state(y, predictions)
        return loss_value

    def fit(
        self,
        x=None,
        y=None,
        batch_size=None,
        epochs=1,
        verbose="auto",
        callbacks=None,
    ):
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        loss_value = 0
        logs = {}
        if callbacks is not None:
            self.callbacks = keras.callbacks.CallbackList(callbacks=callbacks, model=self, verbose=True, epochs=5,)
        #for n in callbacks:
        #    self.callbacks.append(n)

        # Training loop
        for epoch in range(epochs):
            start_time = time.time()
            logs["epoch"] = epoch
            self.callbacks.on_epoch_begin(epoch=epoch, logs=logs)

            # loss_value = self.train_step(x, y)

            # Loop over the batches of the dataset
            # for step, (x_batch_train, y_batch_train) in enumerate(dataset):
            for batch in range(0, 200, 1):
                self.callbacks.on_batch_begin(batch, logs)
                x_batch_train, y_batch_train = x[batch:batch_size + batch], y[batch:batch_size + batch]
                # Compute a training step
                loss_value = self.train_step(x_batch_train, y_batch_train)
                self.callbacks.on_batch_end(batch, logs)

            # Log the loss value
            print('Epoch {}: loss = {}'.format(epoch, loss_value))

            self.callbacks.on_epoch_end(epoch, logs)


class GanModel(keras.Sequential):
    def __init__(self, **kwargs):
        super().__init__(self, **kwargs)








"""def fit(
        self,
        x=None,
        y=None,
        batch_size=None,
        epochs=1,
        verbose="auto",
        callbacks=None,
        validation_split=0.0,
        validation_data=None,
        shuffle=True,
        class_weight=None,
        sample_weight=None,
        initial_epoch=0,
        steps_per_epoch=None,
        validation_steps=None,
        validation_batch_size=None,
        validation_freq=1,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
    ):"""

"""model = keras.Sequential()
        model.add(layers.LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
        model.add(layers.LSTM(100, return_sequences=False))
        model.add(layers.Dense(25))
        model.add(layers.Dense(1))
        model.summary()

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00003), loss='mean_squared_error')
"""