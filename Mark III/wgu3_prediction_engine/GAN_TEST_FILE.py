import multiprocessing as mp
import sys

from gan import gan
from gan.customModels import customModels

if __name__ == '__main__':
    mp.freeze_support()
    run_mode = 'train'

    # Setup queues for inter-process communications
    display_queue = mp.Queue()
    progress_queue = mp.Queue()
    time_queue = mp.Queue()
    scaler_queue = mp.Queue()
    state_queue = mp.Queue()
    update_event = mp.Event()
    epoch_event = mp.Event()
    batch_event = mp.Event()

    queues = {
        'batch_prog_queue': progress_queue,
        'display_queue': display_queue,
        'time_queue': time_queue,
        'scaler_queue': scaler_queue,
        'state_queue': state_queue,
        'update_event': update_event,
        'epoch_event': epoch_event,
        'batch_event': batch_event,
    }

    match run_mode:
        case 'train':
            gan_thread = mp.Process(target=gan.train, args=(queues,), daemon=True)
            gan_thread.start()

            gui = customModels.TrainingGui(queues)
            # gui.start_gui()
        case 'run':
            gan_thread = mp.Process(target=gan.run)
            gan_thread.start()
        case _:
            print("\nNo valid run_mode. Specify a valid run_mode at the start of the main function.\nUse one"
                  " of the following options:   'train'\n                                    'run'")
            sys.exit(-1)

    sys.exit(0)
