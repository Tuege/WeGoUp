import sys
import logging
import threading
import time


class PositionManager:
    def __init__(self):
        print("Position Manager module .... instantiated")
        self.current_prediction = None
        self.current_strategy_list = None
        self.local_exit_event = threading.Event()

    def get_prediction(self, prediction_lock):
        with prediction_lock:
            #TODO: add the correct variable for prediction
            self.current_prediction = None
            print("Position Manager: Prediction Data successfully retrieved")

    def get_strategy_list(self, strategy_lock):
        with strategy_lock:
            # TODO: add the correct variable for strategy list
            current_strategy_list = None
            print("Position Manager: Strategy List successfully retrieved")

    def pass_event(self, exit_event):
        self.local_exit_event = exit_event

    def run(self, sequential_lock):
        while True:

            for a in range(100):
                for n in range(100000):
                    n = n * n

            with sequential_lock:
                print("0: Starting Position Manager")
                print("0: Closing Position Manager")

            if self.local_exit_event.is_set():
                return


class PositionManagerBT(PositionManager):
    def __init__(self):
        print("Position Manager module instantiated --- Backtesting enabled ---")

