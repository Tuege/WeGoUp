import multiprocessing as mp
import threading
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import scipy.stats as stats
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
error_stats_list = []
ani = None

def onclick(event, ax):
    '''
    Event handler for button_press_event
    @param event MouseEvent
    '''
    global ix
    ix = event.xdata
    iy = event.ydata

    ax_rmse, ax_histogram, ax_progress, ax_prediction = ax[:]

    for i, a in enumerate(ax):

        # For information, print which axes the click was in
        if ax == event.inaxes:
            print("Click is in axes ax{}".format(i+1))

    # Check if the click was in ax4 or ax5
    if event.inaxes is ax_rmse:

        if ix is not None:
            print('x = %f' %(ix))
            print('y = %f' %(iy))

        match ax_rmse.get_yscale():
            case 'linear':
                ax_rmse.set_yscale('log')
            case 'log':
                ax_rmse.set_yscale('linear')
        return ix, iy

    else:
        return

def scan(disp_queue: mp.Queue, prog_queue: mp.Queue, tim_queue: mp.Queue, ax):
    last_tim = time.time()
    while 1:
        interval = time.time() - last_tim
        if not prog_queue.empty() or not disp_queue.empty() or not tim_queue.empty() or interval >= 1:
            last_tim = time.time()
            ani.resume()
            while not prog_queue.empty() or not disp_queue.empty() or not tim_queue.empty() or interval > 1:
                interval = time.time() - last_tim

def update(frame, disp_queue: mp.Queue, prog_queue: mp.Queue, tim_queue: mp.Queue, ax):
    global ani, start_run_time, epoch_bars, batch_bars, current_time_text, batch_time_text
    ax_rmse, ax_histogram, ax_progress, ax_prediction = ax[:]

    if not prog_queue.empty():
        pgr_epoch, pgr_batch = prog_queue.get()
        #ax_progress.clear()
        ax_progress.axis('off')

        size = 0.05
        progress_angle = (pgr_epoch[0] / pgr_epoch[1]) * 360
        vals_rad = np.empty(int(pgr_epoch[0] / pgr_epoch[2]))
        vals_rad.fill(np.radians(-(pgr_epoch[2] / pgr_epoch[1]) * 360))
        vals = np.cumsum(np.append((90 / 360) * 2 * np.pi, vals_rad[:-1]))
        try:
            epoch_bars.remove()
        except:
            pass
        epoch_bars = ax_progress.bar(x=vals,
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
        try:
            batch_bars.remove()
        except:
            pass
        batch_bars = ax_progress.bar(x=vals,
                        width=vals_rad, bottom=0.8 - 2 * size, height=size,
                        edgecolor='orange', color='orange', linewidth=0.5, align="edge")
        angle = progress_angle
        #ax_progress.text((-(angle - 90) / 360) * 2 * np.pi, 0.1, 'Batch', fontsize=9, color='orange', ha='center', va='bottom', rotation=-(angle - 90))

    if not disp_queue.empty():
        rmse, target, prediction, error, batch_time_list = disp_queue.get()
        old_x_lim = ax_rmse.get_xlim()
        old_y_lim = ax_rmse.get_ylim()
        old_scale = ax_rmse.get_yscale()
        if len(rmse) > 2 and (((old_x_lim[0] > 0) or (old_x_lim[1] < len(rmse[:-1])-1)) or ((old_y_lim[0] > min(rmse[:-1])) or (old_y_lim[1] < max(rmse[:-1])))):
            ax_rmse.clear()
            ax_rmse.set_xlim(old_x_lim)
            ax_rmse.set_ylim(old_y_lim)
        else:
            ax_rmse.clear()
        match old_scale:
            case 'linear':
                ax_rmse.set_yscale('linear')
            case 'log':
                ax_rmse.set_yscale('log')
        ax_rmse.plot(rmse, color='#4a8fdd')
        #figure = plt.gcf()
        #toolbar = figure.canvas.toolbar
        #toolbar.update()

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
        ax_prediction.plot(target, color='#c75450')
        prediction_list.append(prediction)
        for n in range(len(prediction_list)):
            ax_prediction.plot(prediction_list[n], color='#4a8fdd', alpha=(1/(len(prediction_list)-n)))

        ax_prediction.set_ylim([60, 170])
        ax_prediction.set_ylabel('Price ($)')
        ax_prediction.set_xlabel('Date')

        ax_rmse.set_ylabel('RMSE')
        # ax_error.set_ylabel('Total Error')

        ax_histogram.clear()
        diff = target-prediction
        counts, bins = np.histogram(diff, bins='auto')
        ax_histogram.hist(bins[:-1], bins, weights=counts*(5/len(prediction)))
        mu = np.mean(diff)
        sigma = np.std(diff)
        error_stats_list.append([mu, sigma])
        print(len(error_stats_list))
        for n in range(len(error_stats_list)):
            print((1/(len(error_stats_list)-n)))
            mu, sigma = error_stats_list[n]
            x = np.linspace(mu - 10 * sigma, mu + 10 * sigma, 100)
            ax_histogram.plot(x, abs(mu) * 0.3 * stats.norm.pdf(x, mu, sigma), color='#4a8fdd', alpha=(1/(len(error_stats_list)-n)))


    if not tim_queue.empty():
        batch_time_list = tim_queue.get()
        seconds = np.mean(batch_time_list)
        seconds = seconds % (24 * 3600)
        hour = seconds // 3600
        seconds %= 3600
        minutes = seconds // 60
        seconds %= 60
        try:
            batch_time_text.remove()
        except:
            pass
        if hour != 0:
            batch_time_text = ax_progress.text(np.radians(45), 1,
                                               'Avg. Batch:\n' + str(int(hour)) + 'h ' + str(int(minutes)) + 'm ' + str(
                                                   round(seconds,1)) + 's', ha='right', color='#747a80', size=9)
        elif minutes != 0:
            batch_time_text = ax_progress.text(np.radians(45), 1,
                                               'Avg. Batch:\n' + str(int(minutes)) + 'm ' + str(round(seconds,1)) + 's', ha= 'right', color='#747a80', size=9)
        else:
            batch_time_text = ax_progress.text(np.radians(45), 1, 'Avg. Batch:\n' + str(round(seconds,1)) + 's', ha='right', color='#747a80', size=9)
        print("Batch Time Update")

    seconds = round(time.time() - start_run_time, 0)
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    try:
        current_time_text.remove()
    except:
        pass
    if hour != 0:
        current_time_text = ax_progress.text(np.radians(135), 1,
                                             'Run Time:\n' + str(int(hour)) + 'h ' + str(int(minutes)) + 'm ' + str(
                                                 int(seconds)) + 's', color='#747a80', size=9)
    elif minutes != 0:
        current_time_text = ax_progress.text(np.radians(135), 1,
                                             'Run Time:\n' + str(int(minutes)) + 'm ' + str(int(seconds)) + 's', color='#747a80', size=9)
    else:
        current_time_text = ax_progress.text(np.radians(135), 1, 'Run Time:\n' + str(int(seconds)) + 's', color='#747a80', size=9)

    ani.pause()


if __name__ == '__main__':
    global start_run_time
    mp.freeze_support()
    run_mode = 'train'

    display_queue = mp.Queue()
    progress_queue = mp.Queue()
    time_queue = mp.Queue()
    e = mp.Event()

    match run_mode:
        case 'train':
            gan_thread = mp.Process(target=gan.train, args=(display_queue, progress_queue, time_queue), daemon=True)
            gan_thread.start()
            start_run_time = time.time()

            fig = plt.figure(constrained_layout=True)
            gs = GridSpec(2, 3, figure=fig)

            axs = []
            axs.append(fig.add_subplot(gs[0, 0]))
            # axs.append(axs[0].twinx())
            axs.append(fig.add_subplot(gs[0, 1]))
            axs.append(fig.add_subplot(gs[0, 2], projection='polar'))
            axs.append(fig.add_subplot(gs[1, :]))

            cid = fig.canvas.mpl_connect('button_press_event', lambda event: onclick(event, axs))

            match 'beautify':
                case 'beautify':
                    fig.set_facecolor('#2b2b2b')
                    for n in axs:
                        n.set_facecolor('#3c3f41')
                        n.spines[:].set_color('#747a80')
                        n.tick_params(axis='both', which='both', colors='#747a80', labelcolor='#747a80')
                        n.xaxis.label.set_color('#747a80')
                        n.yaxis.label.set_color('#747a80')
                        n.title.set_color('#747a80')
                case _:
                    pass


            update_thread = threading.Thread(target=scan, args=(display_queue, progress_queue, time_queue, axs), daemon=True)
            update_thread.start()

            ani = animation.FuncAnimation(fig, update, fargs=(display_queue, progress_queue, time_queue, axs))
            plt.show()

        case 'run':
            gan_thread = mp.Process(target=gan.run)
            gan_thread.start()
        case _:
            print("\nNo valid run_mode. Specify a valid run_mode at the start of the main function.\nUse one"
                  " of the following options:   'train'\n                                    'run'")
            sys.exit(-1)

    sys.exit(0)