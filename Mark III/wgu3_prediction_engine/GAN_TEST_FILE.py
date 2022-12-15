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

def run_gan():
    gan.run()


prediction_list = []
line_list = []


def update(frame, disp_queue: mp.Queue, ax):
    ax_rmse, ax_prediction = ax[:]
    rmse, target, prediction, error = disp_queue.get()
    ax_rmse.clear()
    ax_rmse.plot(rmse, color='#4a8fdd')
    #ax_error.clear()
    #ax_error.plot(error, color='#4a8fdd')
    ax_prediction.clear()
    ax_prediction.plot(target, color='#da0f20')
    prediction_list.append(prediction)
    for n in range(len(prediction_list)):
        ax_prediction.plot(prediction_list[n], color='#4a8fdd', alpha=(1/(len(prediction_list)-n)))

    ax_prediction.set_ylim([60, 170])
    ax_prediction.set_ylabel('Price ($)')
    ax_prediction.set_xlabel('Date')

    ax_rmse.set_ylabel('RMSE')
    #ax_error.set_ylabel('Total Error')



if __name__ == '__main__':
    mp.freeze_support()
    run_mode = 'run'

    display_queue = mp.Queue()

    match run_mode:
        case 'train':
            gan_thread = mp.Process(target=gan.train, args=(display_queue,))
        case 'run':
            gan_thread = mp.Process(target=gan.run)
        case _:
            print("\nNo valid run_mode. Specify a valid run_mode at the start of the main function.\nUse one"
                  " of the following options:   'train'\n                                    'run'")
            sys.exit(-1)
    gan_thread.start()

    fig, axes = plt.subplots(ncols=2)
    ani = animation.FuncAnimation(fig, update, fargs=(display_queue, axes))
    plt.show()

    gan_thread.join()
    ani.pause()
    #sys.exit(0)


