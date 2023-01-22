import multiprocessing as mp
import threading
import time
import random
import sys
import os

import matplotlib.pyplot as plt
import matplotlib.animation as animation
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
        # self.fig = plt.figure()
        self.fig, self.ax = plt.subplots()
        self.queues = queues

        # self.update_process = mp.Process(target=self.update_loop)
        self.scan_thread = threading.Thread(target=self.scan)
        self.counter = 0
        self.running = False

    def on_close(self, event):
        # print('Closed Figure!')
        # self.update_process.terminate()
        # sys.exit(0)
        # raise SystemExit
        pass

    def draw_event_callback(self, event):
        print("Draw Event Triggered!")

    def onclick(self, event):
        '''
        Event handler for button_press_event
        @param event MouseEvent
        '''
        global ix
        ix = event.xdata
        iy = event.ydata

        ax = [self.ax]

        for i, a in enumerate(ax):

            # For information, print which axes the click was in
            if a == event.inaxes:
                print("Click is in axes ax{}".format(i + 1))

        # Check if the click was in ax4 or ax5
        if event.inaxes is a:

            if ix is not None:
                print('x = %f' % (ix))
                print('y = %f' % (iy))

            match a.get_yscale():
                case 'linear':
                    a.set_yscale('log')
                case 'log':
                    a.set_yscale('linear')
            self.ani.resume()
            return ix, iy

        else:
            return

    def start_gui(self):
        self.ani = animation.FuncAnimation(self.fig, self.update_function)

        self.scan_thread.start()

        # self.update_process.start()
        # cid = self.fig.canvas.mpl_connect('close_event', self.on_close)
        # bid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        # did = self.fig.canvas.mpl_connect('draw_event', self.draw_event_callback)

        plt.text(0.35, 0.5, 'Close Me!', dict(size=30))
        plt.show()

    def scan(self):
        try:
            bid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
            while True:
                while not self.queues['update_event'].is_set():
                    pass
                self.queues['update_event'].clear()
                self.counter += 1
                self.ax.set_title(str(self.counter))
                if self.queues['epoch_event']:
                    self.ax.set_xlabel(str("Epoch " + str(queues['state_queue'].get()['epoch'])))
                if self.queues['batch_event']:
                    self.ax.set_ylabel(str("Batch " + str(queues['state_queue'].get()['batch'])))
                while self.running:
                    pass
                self.ani.resume()
        finally:
            return

    def update_function(self, frame):
        # self.running = True
        # self.running = False
        self.ani.pause()


def random_feeder(queues):
    epoch = 0
    batch = 0
    while True:
        sleep_time = float(random.randint(1, 1000))/float(1000)
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
