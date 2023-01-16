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

        logs['epochs'] = epochs
        logs['batches'] = batch_size
        self.callbacks.on_train_begin(logs)
        del logs['epochs']
        del logs['batches']

        # self.reset_metrics()
        self.compiled_metrics.update_state(None, None)
        for m in self.metrics:
            if m.name == 'epoch':
                m.update_epoch(0)
            elif m.name == 'epochs':
                m.update_epochs(epochs)
            elif m.name == 'batches':
                m.update_batches(np.shape(x)[0]-(batch_size-1))

        # Training loop
        for epoch in range(epochs):
            start_time = time.time()
            self.reset_metrics()
            for m in self.metrics:
                if m.name == 'epoch':
                    m.update_epoch(epoch + 1)
                if m.name == 'batch':
                    m.update_batch(0)
            self.callbacks.on_epoch_begin(epoch, logs)

            # Loop over the batches of the dataset
            # for step, (x_batch_train, y_batch_train) in enumerate(dataset):
            for batch in range(0, np.shape(x)[0]-(batch_size-1), 1):
                self.callbacks.on_batch_begin(batch, logs)
                x_batch_train, y_batch_train = x[batch:batch_size + batch], y[batch:batch_size + batch]
                # Compute a training step
                logs = self.train_step(x_batch_train, y_batch_train)
                for m in self.metrics:
                    if m.name == 'batch':
                        m.update_batch(batch + 1)
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

    """
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

        def on_batch_end(self, batch, logs=None):
            pass
            epoch, epochs = logs["epoch", "epochs"]
            self.queues['batch_prog_queue'].put([[epoch + 1, epochs, 1], [batch + 1, 200, 1]])
            # print('batch', batch+1)
    """

    class CustomCallbacks:
        class Scheduler(keras.callbacks.LearningRateScheduler):
            def __init__(self):
                super().__init__(schedule=self.scheduler, verbose=True)

            def scheduler(self, epoch, lr):
                if epoch < 1:
                    return lr
                else:
                    return lr * 1
                    # print(lr * tf.math.exp(-0.1))
                    # return lr * tf.math.exp(-0.1)

        class Progress(keras.callbacks.Callback):
            def __init__(self, queues=None):
                super().__init__()
                self.queues = queues

            def on_train_begin(self, logs=None):
                self.queues['state_queue'].put({
                    'loss': 0,
                    'epoch': 0,
                    'epochs': logs['epochs'],
                    'batch': 0,
                    'batches': logs['batches'],
                })

            def on_epoch_end(self, epoch, logs=None):
                pass
                # print("Epoch has ended")

            def on_batch_begin(self, batch, logs=None):
                state = self.queues['state_queue'].get()
                state['batch'] = batch
                self.queues['state_queue'].put(state)

            def on_batch_end(self, batch, logs=None):
                pass
                # self.queues['batch_prog_queue'].put([[epoch + 1, epochs, 1], [batch + 1, 200, 1]])
                # print('batch', batch+1)

    class CustomMetrics:
        class _CustomMetric(keras.metrics.Metric):
            def __init__(self, name='metric', **kwargs):
                super().__init__(name=name, **kwargs)
                self.metric = 0

            def result(self):
                return self.metric

            def reset_state(self):
                self.metric = self.metric
                # pass

        class Epoch(_CustomMetric):
            def __init__(self, name='epoch', **kwargs):
                super().__init__(name=name, **kwargs)

            def update_state(self, *args, **kwargs):
                self.update_epoch()

            def update_epoch(self, epoch=None):
                if epoch is not None:
                    self.metric = epoch

        class Epochs(_CustomMetric):
            def __init__(self, name='epochs', **kwargs):
                super().__init__(name=name, **kwargs)

            def update_state(self, *args, **kwargs):
                self.update_epochs()

            def update_epochs(self, epochs=None):
                if epochs is not None:
                    self.metric = epochs

        class Loss(_CustomMetric):
            def __init__(self, name='loss', **kwargs):
                super().__init__(name=name, **kwargs)

            def update_state(self, *args, **kwargs):
                self.update_loss()

            def update_loss(self, loss=None):
                if loss is not None:
                    self.metric = loss

        class Batch(_CustomMetric):
            def __init__(self, name='batch', **kwargs):
                super().__init__(name=name, **kwargs)

            def update_state(self, *args, **kwargs):
                self.update_batch()

            def update_batch(self, batch=None):
                if batch is not None:
                    self.metric = batch

        class Batches(_CustomMetric):
            def __init__(self, name='batches', **kwargs):
                super().__init__(name=name, **kwargs)

            def update_state(self, *args, **kwargs):
                self.update_batches()

            def update_batches(self, batches=None):
                if batches is not None:
                    self.metric = batches

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