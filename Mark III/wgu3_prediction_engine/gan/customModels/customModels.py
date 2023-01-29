import tensorflow as tf
from tensorflow import keras
from keras import layers
import sys
import time
import numpy as np
import scipy.stats as stats
import threading
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec


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
    ):
        # dataset = tf.data.Dataset.from_tensor_slices((x, y))
        logs = {}
        # if callbacks is not None:
        self.callbacks = keras.callbacks.CallbackList(callbacks=callbacks, model=self, verbose=True, epochs=epochs,)
        self.stop_training = False

        print('Input shape: ', np.shape(x))

        # self.reset_metrics()
        self.compiled_metrics.update_state(None, None)
        for m in self.metrics:
            if m.name == 'epoch':
                m.update_epoch(0)
            elif m.name == 'epochs':
                m.update_epochs(epochs)
            elif m.name == 'batches':
                m.update_batches(np.shape(x)[0]-(batch_size-1))

        logs['epochs'] = epochs
        logs['batches'] = np.shape(x)[0]-(batch_size-1)
        logs['target'] = y
        self.callbacks.on_train_begin(logs)

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
            # for m in self.metrics:
            #     print('    {}: {}'.format(m.name, m.result()))

            logs['prediction'] = self(x, training=False)
            self.callbacks.on_epoch_end(epoch, logs)
            if self.stop_training:
                break

    class CustomCallbacks:
        class Scheduler(keras.callbacks.LearningRateScheduler):
            def __init__(self):
                super().__init__(schedule=self.scheduler, verbose=False)

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
                self.epoch_start_time = 0
                self.batch_start_time = 0

            def on_train_begin(self, logs=None):
                self.queues['state_queue'].put({
                    'loss': 0,
                    'epoch': 0,
                    'epochs': logs['epochs'],
                    'batch': 0,
                    'batches': logs['batches'],
                    'start_time': time.time(),
                    'epoch_time': 0,
                    'batch_time': 0,
                    'target': logs['target'],
                    'prediction': [],
                })
                self.queues['update_event'].set()
                print('\n---Training Started---\n')

            def on_epoch_begin(self, epoch, logs=None):
                self.epoch_start_time = time.time()

            def on_epoch_end(self, epoch, logs=None):
                state = self.queues['state_queue'].get()
                state['epoch'] = epoch + 1
                state['loss'] = logs['loss']
                state['epoch_time'] = time.time() - self.epoch_start_time
                state['prediction'] = logs['prediction']
                self.queues['state_queue'].put(state)
                self.queues['update_event'].set()
                self.queues['epoch_event'].set()

            def on_batch_begin(self, batch, logs=None):
                self.batch_start_time = time.time()

            def on_batch_end(self, batch, logs=None):
                state = self.queues['state_queue'].get()
                state['batch'] = batch + 1
                state['loss'] = logs['loss']
                state['batch_time'] = time.time() - self.batch_start_time
                self.queues['state_queue'].put(state)
                self.queues['update_event'].set()
                self.queues['batch_event'].set()
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


class TrainingGui:
    def __init__(self, queues):
        self.queues = queues
        self.current_time_text = None
        self.epoch_bar = None
        self.batch_bar = None
        self.prediction_list = []
        self.rmse_list = np.array([])
        self.error_stats_list = []
        while queues['scaler_queue'].empty():
            pass
        self.scaler_list = queues['scaler_queue'].get()

        self.fig = plt.figure(constrained_layout=True)
        gs = GridSpec(2, 3, figure=self.fig)

        self.axs = []
        self.axs.append(self.fig.add_subplot(gs[0, 0]))
        self.axs.append(self.fig.add_subplot(gs[0, 1]))
        self.axs.append(self.fig.add_subplot(gs[0, 2], projection='polar'))
        self.axs.append(self.fig.add_subplot(gs[1, :]))

        self.ax_rmse, self.ax_histogram, self.ax_progress, self.ax_prediction = self.axs[:]

        match 'beautify':
            case 'beautify':
                self.fig.set_facecolor('#2b2b2b')
                for n in self.axs:
                    n.set_facecolor('#3c3f41')
                    n.spines[:].set_color('#747a80')
                    n.tick_params(axis='both', which='both', colors='#747a80', labelcolor='#747a80')
                    n.xaxis.label.set_color('#747a80')
                    n.yaxis.label.set_color('#747a80')
                    n.title.set_color('#747a80')
                self.ax_progress.axis('off')
            case _:
                pass

        self.scan_thread = threading.Thread(target=self.scan)

        self.start_gui()

    def on_close(self, event):
        # print('Closed Figure!')
        # self.update_process.terminate()
        # sys.exit(0)
        # raise SystemExit
        pass

    def draw_event_callback(self, event):
        print("Draw Event Triggered!")

    def onclick(self, event):
        ix = event.xdata
        iy = event.ydata

        """for i, a in enumerate(self.axs):

            # For information, print which axes the click was in
            if a == event.inaxes:
                print("Click is in axes ax{}".format(i + 1))"""

        # Check if the click was in ax4 or ax5
        if event.inaxes is self.ax_rmse:

            """if ix is not None:
                print('x = %f' % (ix))
                print('y = %f' % (iy))"""

            match self.ax_rmse.get_yscale():
                case 'linear':
                    self.ax_rmse.set_yscale('log')
                case 'log':
                    self.ax_rmse.set_yscale('linear')
            self.fig.canvas.draw()
            return ix, iy

        else:
            return

    def start_gui(self):
        # self.ani = animation.FuncAnimation(self.fig, self.redraw)

        self.scan_thread.start()

        # cid = self.fig.canvas.mpl_connect('close_event', self.on_close)
        bid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        # did = self.fig.canvas.mpl_connect('draw_event', self.draw_event_callback)

        # plt.text(0.35, 0.5, 'Close Me!', dict(size=30))

        plt.show()

    def scan(self):
        try:
            while True:
                # while not self.queues['update_event'].is_set():
                #     pass
                self.queues['update_event'].wait()
                self.queues['update_event'].clear()
                self.update_function()
        finally:
            # This is required to automatically shutdown this thread when the figure is closed
            return

    def update_function(self):
        state = self.queues['state_queue'].get()
        self.queues['state_queue'].put(state)

        if self.queues['epoch_event'].is_set():
            self.update_epoch(state)
            self.queues['epoch_event'].clear()
        if self.queues['batch_event'].is_set():
            self.update_batch(state)
            self.queues['batch_event'].clear()

        self.update_time(state)

        self.fig.canvas.draw()
        # self.ani.resume()

    def update_epoch(self, state):
        # Progress Bar Axis
        size = 0.05
        progress_angle = (state['epoch'] / state['epochs']) * 360
        val = -np.radians(progress_angle)

        if self.epoch_bar is not None:
            self.epoch_bar.remove()
        self.epoch_bar = self.ax_progress.bar(x=np.radians(90),
                                              width=val, bottom=0.65 - 2 * size, height=size,
                                              edgecolor='w', color='#ffb600', linewidth=0, align="edge")

        # Prediction Axis
        self.ax_prediction.clear()
        target = np.ravel(self.scaler_list[0].inverse_transform(state['target'].reshape(-1, 1)))
        self.ax_prediction.plot(target, color='#c75450')
        prediction = np.ravel(self.scaler_list[0].inverse_transform(state['prediction'].reshape(-1, 1)))
        self.prediction_list.append(prediction)
        for n in range(len(self.prediction_list)):
            self.ax_prediction.plot(self.prediction_list[n], color='#4a8fdd',
                                    alpha=(1 / (len(self.prediction_list) - n)))

        self.ax_prediction.set_ylim([0, 170]) #60
        self.ax_prediction.set_ylabel('Price ($)')
        self.ax_prediction.set_xlabel('Date')
        self.fig.canvas.draw()

        # RMSE Axis
        old_x_lim = self.ax_rmse.get_xlim()
        old_y_lim = self.ax_rmse.get_ylim()
        old_scale = self.ax_rmse.get_yscale()
        print("getting stuck at the scaler conversion of the loss")
        np.append(self.rmse_list, state['loss'])
        self.rmse_list = np.ravel(self.scaler_list[0].inverse_transform(self.rmse_list.reshape(-1, 1)))
        print("getting past the scaler conversion of the loss")
        if len(self.rmse_list) > 2 and (((old_x_lim[0] > 0) or (old_x_lim[1] < len(self.rmse_list[:-1]) - 1)) or (
                (old_y_lim[0] > min(self.rmse_list[:-1])) or (old_y_lim[1] < max(self.rmse_list[:-1])))):
            self.ax_rmse.clear()
            self.ax_rmse.set_xlim(old_x_lim)
            self.ax_rmse.set_ylim(old_y_lim)
        else:
            self.ax_rmse.clear()
        match old_scale:
            case 'linear':
                self.ax_rmse.set_yscale('linear')
            case 'log':
                self.ax_rmse.set_yscale('log')
        self.ax_rmse.plot(self.rmse_list, color='#4a8fdd')
        self.ax_rmse.set_ylabel('RMSE')

        # Histogram Axis
        self.ax_histogram.clear()
        diff = []
        # target = state['target']
        # prediction = state['prediction']
        target = np.ravel(self.scaler_list[0].inverse_transform(state['target'].reshape(-1, 1)))
        prediction = np.ravel(self.scaler_list[0].inverse_transform(state['prediction'].reshape(-1, 1)))
        diff = target - prediction
        # for i in range(len(target)):
        #     diff.append(target[i] - prediction[i])
        counts, bins = np.histogram(diff, bins='auto')
        self.ax_histogram.hist(bins[:-1], bins, weights=counts * (5 / len(state['prediction'])))
        mu = np.mean(diff)
        sigma = np.std(diff)
        self.error_stats_list.append([mu, sigma])
        # print(len(self.error_stats_list))
        for n in range(len(self.error_stats_list)):
            # print((1 / (len(self.error_stats_list) - n)))
            mu, sigma = self.error_stats_list[n]
            x = np.linspace(mu - 10 * sigma, mu + 10 * sigma, 100)
            self.ax_histogram.plot(x, abs(mu) * 0.3 * stats.norm.pdf(x, mu, sigma), color='#4a8fdd',
                                   alpha=(1 / (len(self.error_stats_list) - n)))

        # self.ax_rmse.set_xlabel(str("Epoch " + str(state['epoch'])))

    def update_batch(self, state):
        size = 0.1
        progress_angle = (state['batch'] / state['batches']) * 360
        val = -np.radians(progress_angle)

        if self.batch_bar is not None:
            self.batch_bar.remove()
        self.batch_bar = self.ax_progress.bar(x=np.radians(90),
                                              width=val, bottom=0.8 - 2 * size, height=size,
                                              edgecolor='orange', color='orange', linewidth=0, align="edge")

        # self.ax_rmse.set_ylabel(str("Batch " + str(state['batch'])))

    def update_time(self, state):
        seconds = round(time.time() - state['start_time'], 0)
        seconds = seconds % (24 * 3600)
        hour = seconds // 3600
        seconds %= 3600
        minutes = seconds // 60
        seconds %= 60
        if self.current_time_text is not None:
            self.current_time_text.remove()
        if hour != 0:
            self.current_time_text = self.ax_progress.text(np.radians(135), 1, 'Run Time:\n' + str(int(hour)) + 'h ' + str(int(minutes)) + 'm ' + str(int(seconds)) + 's', color='#747a80', size=9)
        elif minutes != 0:
            self.current_time_text = self.ax_progress.text(np.radians(135), 1, 'Run Time:\n' + str(int(minutes)) + 'm ' + str(int(seconds)) + 's', color='#747a80', size=9)
        else:
            self.current_time_text = self.ax_progress.text(np.radians(135), 1, 'Run Time:\n' + str(int(seconds)) + 's', color='#747a80', size=9)

        self.fig.canvas.draw()

    def redraw(self, frame):
        # self.ani.pause()
        pass









"""model = keras.Sequential()
        model.add(layers.LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
        model.add(layers.LSTM(100, return_sequences=False))
        model.add(layers.Dense(25))
        model.add(layers.Dense(1))
        model.summary()

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00003), loss='mean_squared_error')
"""