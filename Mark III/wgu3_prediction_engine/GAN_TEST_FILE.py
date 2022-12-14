import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from gan import gan
import os
import sys
import time
import importlib

#from DCGAN import dcgan as gan
import DCGAN


def count():
    for n in range(10):
        print(n)
        for i in range(200000000):
            pass


def train_gan(disp_queue: mp.Queue):
    """gan.trainer(params)
    gan.optimiser(params)
    gan.hyperparameter_optimiser(params)
    gan.train(dataset, hyperparameter_optimisation=True)"""
    gan.train(disp_queue)


prediction_list = []
line_list = []


def update(frame, disp_queue: mp.Queue, ax_error, ax_prediction):
    error, target, prediction = disp_queue.get()
    ax_error.clear()
    ax_error.plot(error, color='#4a8fdd')
    ax_prediction.clear()
    ax_prediction.plot(target, color='#da0f20')
    prediction_list.append(prediction)
    for n in range(len(prediction_list)):
        ax_prediction.plot(prediction_list[n], color='#4a8fdd', alpha=(1/(len(prediction_list)-n)))
        #line_list.append(line)

    """if len(prediction_list) == 5:
        prediction_list.pop(0)
    prediction_list.append(prediction)
    for n in range(len(prediction_list)-1):
        ax_prediction.plot(prediction_list[n], color='#e2e2e2')
    ax_prediction.plot(prediction_list[len(prediction_list)-1], color='#4a8fdd')"""


    #ax_error.set_ylim([60, 170])
    ax_prediction.set_ylim([60, 170])


if __name__ == '__main__':
    mp.freeze_support()

    fig, axes = plt.subplots(ncols=2)
    axes[0].set_title('Title')

    display_queue = mp.Queue()
    gan_thread = mp.Process(target=gan.train, args=(display_queue,))
    # other_thread = mp.Process(target=count)
    gan_thread.start()
    # other_thread.start()

    ani = animation.FuncAnimation(fig, update, fargs=(display_queue, axes[0], axes[1]))  # blit=True,
    plt.show()

    gan_thread.join()

    sys.exit(0)


