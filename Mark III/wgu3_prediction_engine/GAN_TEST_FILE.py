import multiprocessing as mp
import threading
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import numpy as np
from gan import gan
import os
import sys
import time
import importlib

#from DCGAN import dcgan as gan
import DCGAN


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
ani = None


def scan(disp_queue: mp.Queue, prog_queue: mp.Queue, ax):
    while 1:
        if not prog_queue.empty() or not disp_queue.empty():
            ani.resume()
            while not prog_queue.empty() or not disp_queue.empty():
                pass


def update(frame, disp_queue: mp.Queue, prog_queue: mp.Queue, ax):
    global ani
    ax_rmse, ax_rmse_log, ax_progress, ax_prediction = ax[:]

    if not prog_queue.empty():

        pgr_epoch, pgr_batch = prog_queue.get()
        ax_progress.clear()
        ax_progress.axis('off')

        size = 0.05
        progress_angle = (pgr_epoch[0] / pgr_epoch[1]) * 360
        vals_rad = np.empty(int(pgr_epoch[0] / pgr_epoch[2]))
        vals_rad.fill(np.radians(-(pgr_epoch[2] / pgr_epoch[1]) * 360))
        vals = np.cumsum(np.append((90 / 360) * 2 * np.pi, vals_rad[:-1]))
        # vals = np.radians([90,86.4,82.8,79.2,75.6,72,68.4,64.8,61.2,57.6])
        # vals_rad = np.radians([3.6,3.6,3.6,3.6,3.6,3.6,3.6,3.6,3.6,3.6])
        ax_progress.bar(x=vals,
                        width=vals_rad, bottom=0.65 - 2 * size, height=size,
                        edgecolor='w', color='#ffb600', linewidth=0, align="edge")
        angle = progress_angle
        # ax_progress.text((-(angle - 90) / 360) * 2 * np.pi, 0.1, 'Epoch', fontsize=9, color='orange', ha='center', va='bottom', rotation=-(angle - 90))

        size = 0.1
        progress_angle = (pgr_batch[0] / pgr_batch[1]) * 360
        vals_rad = np.empty(int(pgr_batch[0] / pgr_batch[2]))
        vals_rad.fill(np.radians(-(pgr_batch[2] / pgr_batch[1]) * 360))
        vals = np.cumsum(np.append((90 / 360) * 2 * np.pi, vals_rad[:-1]))
        # vals = np.radians([90,86.4,82.8,79.2,75.6,72,68.4,64.8,61.2,57.6])
        # vals_rad = np.radians([3.6,3.6,3.6,3.6,3.6,3.6,3.6,3.6,3.6,3.6])
        ax_progress.bar(x=vals,
                        width=vals_rad, bottom=0.8 - 2 * size, height=size,
                        edgecolor='orange', color='orange', linewidth=0.5, align="edge")
        angle = progress_angle
        #ax_progress.text((-(angle - 90) / 360) * 2 * np.pi, 0.1, 'Batch', fontsize=9, color='orange', ha='center', va='bottom', rotation=-(angle - 90))

    if not disp_queue.empty():
        rmse, target, prediction, error = disp_queue.get()
        old_x_lim = ax_rmse.get_xlim()
        old_y_lim = ax_rmse.get_ylim()
        if len(rmse) > 2 and (((old_x_lim[0] > 0) or (old_x_lim[1] < len(rmse[:-1])-1)) or ((old_y_lim[0] > min(rmse[:-1])) or (old_y_lim[1] < max(rmse[:-1])))):
            ax_rmse.clear()
            ax_rmse.set_xlim(old_x_lim)
            ax_rmse.set_ylim(old_y_lim)
        else:
            ax_rmse.clear()
        ax_rmse.plot(rmse, color='#4a8fdd')
        #figure = plt.gcf()
        #toolbar = figure.canvas.toolbar
        #toolbar.update()

        old_x_lim = ax_rmse_log.get_xlim()
        old_y_lim = ax_rmse_log.get_ylim()
        """if len(rmse) > 2 and (((old_x_lim[0] > 0) or (old_x_lim[1] < len(rmse[:-1])-1)) or ((old_y_lim[0] > min(rmse[:-1])) or (old_y_lim[1] < max(rmse[:-1])))):
            ax_rmse_log.clear()
            ax_rmse.set_xlim(old_x_lim)
            ax_rmse.set_ylim(old_y_lim)
        else:
            ax_rmse_log.clear()
        ax_rmse_log.plot(rmse, color='#4a8fdd')
        ax_rmse_log.set_yscale('log')"""

        # ax_error.clear()
        # ax_error.plot(error, color='#4a8fdd')
        ax_prediction.clear()
        ax_prediction.plot(target, color='#c75450')#da0f20')
        prediction_list.append(prediction)
        for n in range(len(prediction_list)):
            ax_prediction.plot(prediction_list[n], color='#4a8fdd', alpha=(1/(len(prediction_list)-n)))

        ax_prediction.set_ylim([60, 170])
        ax_prediction.set_ylabel('Price ($)')
        ax_prediction.set_xlabel('Date')

        ax_rmse.set_ylabel('RMSE')
        # ax_error.set_ylabel('Total Error')

    ani.pause()


if __name__ == '__main__':
    mp.freeze_support()
    run_mode = 'train'

    display_queue = mp.Queue()
    progress_queue = mp.Queue()
    e = mp.Event()

    match run_mode:
        case 'train':
            gan_thread = mp.Process(target=gan.train, args=(display_queue, progress_queue), daemon=True)
            gan_thread.start()

            fig = plt.figure(constrained_layout=True)
            gs = GridSpec(2, 2, figure=fig)

            axs = []
            axs.append(fig.add_subplot(gs[0, 0]))
            axs.append(axs[0].twinx())
            axs.append(fig.add_subplot(gs[0, 1], projection='polar'))
            axs.append(fig.add_subplot(gs[1, :]))

            match 'None':
                case 'beautify':
                    fig.set_facecolor('#2b2b2b')
                    for n in axs:
                        n.set_facecolor('#3c3f41')
                        n.spines[:].set_color('#747a80')
                        n.tick_params(axis='x', colors='#747a80')
                        n.tick_params(axis='y', colors='#747a80')
                        n.xaxis.label.set_color('#747a80')
                        n.yaxis.label.set_color('#747a80')
                        n.title.set_color('#747a80')
                case _:
                    pass


            update_thread = threading.Thread(target=scan, args=(display_queue, progress_queue, axs), daemon=True)
            update_thread.start()

            ani = animation.FuncAnimation(fig, update, fargs=(display_queue, progress_queue, axs))
            plt.show()

        case 'run':
            gan_thread = mp.Process(target=gan.run)
            gan_thread.start()
        case _:
            print("\nNo valid run_mode. Specify a valid run_mode at the start of the main function.\nUse one"
                  " of the following options:   'train'\n                                    'run'")
            sys.exit(-1)

    sys.exit(0)
