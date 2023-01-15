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

        self.queues = {}

    @tf.function
    def train_step(self, inputs, labels):
        with tf.GradientTape() as tape:
            predictions = self(inputs, training=True)
            loss_value = self.compiled_loss(labels, predictions)
        grads = tape.gradient(loss_value, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        # train_acc_metric.update_state(y, predictions)
        self.compiled_metrics.update_state(labels, predictions)
        # return loss_value

        return {m.name: m.result() for m in self.metrics}

    def fit(
        self,
        x=None,
        y=None,
        batch_size=None,
        epochs=1,
        verbose="auto",
        callbacks=None,
    ):
        # dataset = tf.data.Dataset.from_tensor_slices((x, y))
        logs = {}
        # if callbacks is not None:
        self.callbacks = keras.callbacks.CallbackList(callbacks=callbacks, model=self, verbose=True, epochs=epochs,)
        self.stop_training = False

        print(np.shape(x))

        # Training loop
        for epoch in range(epochs):
            start_time = time.time()
            self.reset_metrics()
            self.callbacks.on_epoch_begin(epoch, logs)

            # Loop over the batches of the dataset
            # for step, (x_batch_train, y_batch_train) in enumerate(dataset):
            for batch in range(0, np.shape(x)[0]-(batch_size-1), 1):
                self.callbacks.on_batch_begin(batch, logs)
                x_batch_train, y_batch_train = x[batch:batch_size + batch], y[batch:batch_size + batch]
                # Compute a training step
                logs = self.train_step(x_batch_train, y_batch_train)
                self.callbacks.on_batch_end(batch=batch, logs=logs)
                if self.stop_training:
                    break

            # Log the loss value
            # print('Epoch {}: loss = {}'.format(epoch+1, loss_value))
            for m in self.metrics:
                print('    {}: {}'.format(m.name, m.result()))

            self.callbacks.on_epoch_end(epoch, logs)
            if self.stop_training:
                break

    class CustomScheduler(keras.callbacks.LearningRateScheduler):
        def __init__(self):
            super().__init__(schedule=self.scheduler, verbose=True)

        def scheduler(self, epoch, lr):
            if epoch < 1:
                return lr
            else:
                return lr * 1
                # print(lr * tf.math.exp(-0.1))
                # return lr * tf.math.exp(-0.1)

    class CustomCallbacks(keras.callbacks.Callback):
        def __init__(self, queues=None):
            super().__init__()
            self.queues = queues

        def on_epoch_end(self, epoch, logs=None):
            pass
            # print("Epoch has ended")
            # TODO: replace queues

        def on_batch_end(self, batch, logs=None):
            pass
            self.queues['batch_prog_queue'].put([[batch + 1, 5, 1], [batch + 1, 200, 1]])
            # print('batch', batch+1)


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