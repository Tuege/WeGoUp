import multiprocessing as mp
import threading
import time
import random
import sys
import os

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import numpy as np


class ProcessPlotter:
    def __init__(self):
        self.x = []
        self.y = []

    def terminate(self):
        plt.close('all')

    def call_back(self):
        while self.pipe.poll():
            command = self.pipe.recv()
            if command is None:
                self.terminate()
                return False
            else:
                self.x.append(command[0])
                self.y.append(command[1])
                self.ax.plot(self.x, self.y, 'ro')
        self.fig.canvas.draw()
        return True

    def __call__(self, pipe):
        print('starting plotter...')

        self.pipe = pipe
        self.fig, self.ax = plt.subplots()
        timer = self.fig.canvas.new_timer(interval=1000)
        timer.add_callback(self.call_back)
        timer.start()

        print('...done')
        plt.show()


class Scanner:
    def __init__(self, queues=None):
        self.plot_pipe, plotter_pipe = mp.Pipe()
        self.queues = queues
        self.plotter = ProcessPlotter()
        self.plot_process = mp.Process(
            target=self.plotter, args=(plotter_pipe,), daemon=True)
        self.plot_process.start()

    def plot(self, finished=False):
        send = self.plot_pipe.send
        if finished:
            send(None)
        else:
            data = np.random.random(2)
            send(data)

    def scan(self):
        while True:
            pass


class GuiClass:
    def __init__(self, queues):
        self.queues = queues

        self.fig = plt.figure(constrained_layout=True)
        gs = GridSpec(2, 3, figure=self.fig)

        self.axs = []
        self.axs.append(self.fig.add_subplot(gs[0, 0]))
        self.axs.append(self.fig.add_subplot(gs[0, 1]))
        self.axs.append(self.fig.add_subplot(gs[0, 2], projection='polar'))
        self.axs.append(self.fig.add_subplot(gs[1, :]))

        # cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)

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
            case _:
                pass

        self.scan_thread = threading.Thread(target=self.scan)

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

        ax_rmse, ax_histogram, ax_progress, ax_prediction = self.axs[:]

        for i, a in enumerate(self.axs):

            # For information, print which axes the click was in
            if a == event.inaxes:
                print("Click is in axes ax{}".format(i + 1))

        # Check if the click was in ax4 or ax5
        if event.inaxes is ax_rmse:

            if ix is not None:
                print('x = %f' % (ix))
                print('y = %f' % (iy))

            match ax_rmse.get_yscale():
                case 'linear':
                    ax_rmse.set_yscale('log')
                case 'log':
                    ax_rmse.set_yscale('linear')
            self.ani.resume()
            return ix, iy

        else:
            return

    def start_gui(self):
        self.ani = animation.FuncAnimation(self.fig, self.redraw)

        self.scan_thread.start()

        # cid = self.fig.canvas.mpl_connect('close_event', self.on_close)
        bid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        # did = self.fig.canvas.mpl_connect('draw_event', self.draw_event_callback)

        plt.text(0.35, 0.5, 'Close Me!', dict(size=30))
        plt.show()

    def scan(self):
        try:
            while True:
                # while not self.queues['update_event'].is_set():
                #     pass
                self.queues['update_event'].wait()
                self.queues['update_event'].clear()
                self.update_function()
                self.ani.resume()
        finally:
            # This is required to automatically shutdown this thread when the figure is closed
            return

    def update_function(self):
        ax_rmse, ax_histogram, ax_progress, ax_prediction = self.axs[:]

        if self.queues['epoch_event']:
            ax_rmse.set_xlabel(str("Epoch " + str(queues['state_queue'].get()['epoch'])))
            queues['epoch_event'].clear()
        if self.queues['batch_event']:
            ax_rmse.set_ylabel(str("Batch " + str(queues['state_queue'].get()['batch'])))
            queues['batch_event'].clear()

    def redraw(self, frame):
        self.ani.pause()


def random_feeder(queues):
    epoch = 0
    batch = 0
    while True:
        sleep_time = float(random.randint(1000, 10000))/float(1000)
        time.sleep(sleep_time)
        match random.randint(1, 2):
            case 1:
                queues['update_event'].set()
                queues['epoch_event'].set()
                epoch += 1
                state = {"epoch": epoch, "batch": batch}
                queues['state_queue'].put(state)
            case 2:
                queues['update_event'].set()
                queues['batch_event'].set()
                batch += 1
                state = {"epoch": epoch, "batch": batch}
                queues['state_queue'].put(state)


if __name__ == '__main__':
    display_queue = mp.Queue()
    progress_queue = mp.Queue()
    time_queue = mp.Queue()
    state_queue = mp.Queue()
    update_event = mp.Event()
    epoch_event = mp.Event()
    batch_event = mp.Event()

    queues = {
        'batch_prog_queue': progress_queue,
        'display_queue': display_queue,
        'time_queue': time_queue,
        'state_queue': state_queue,
        'update_event': update_event,
        'epoch_event': epoch_event,
        'batch_event': batch_event,
    }

    random_feeder_process = mp.Process(target=random_feeder, args=(queues,), daemon=True)
    random_feeder_process.start()

    gui = GuiClass(queues)
    gui.start_gui()
