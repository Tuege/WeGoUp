"""
Tuege Neumann
06.11.2022
A GAN based approach to Algorithmic Trading

"""
# %% Imports
import sys
import threading
import atexit

import pandas as pd
import numpy as np
import yfinance as yf

from wgu3_prediction_engine import prediction_engine
from wgu3_strategy_engine import strategy_engine
from wgu3_position_manager import position_manager

# %% Backtesting [enable/disable]
backtesting = True

# %% Module Object Instantiation

if backtesting:
    prediction = prediction_engine.PredictionEngineBT()
    strategy = strategy_engine.StrategyEngineBT()
    position = position_manager.PositionManagerBT()
else:
    prediction = prediction_engine.PredictionEngine()
    strategy = strategy_engine.StrategyEngine()
    position = position_manager.PositionManager()

print("____________________________________________________________________\n")

# %% Threads Setup

'Locks'
stockDataLock = threading.Lock()
technicalIndicatorLock = threading.Lock()
correlatedAssetsLock = threading.Lock()
fourierLock = threading.Lock()
arimaLock = threading.Lock()
predictionLock = threading.Lock()
sequentialLock = threading.Lock()

'Events'
exit_event = threading.Event()

'Threads'
# predictionThread = threading.Thread(target=prediction.run, args=(predictionLock,))
# strategyThread = threading.Thread(target=strategy.run, args=(predictionLock,))
positionThread = threading.Thread(target=position.run, args=(predictionLock,))


def strategy_engine_function():
    t_period = threading.Timer(0.5, strategy_engine_function)
    t_period.start()

    print("1: Prediction Fetched")
    print("1: Opportunities Identified")
    print("1: Feasibility Assessed")
    print("1: Strategy Selected")
    print("1: Orders Passed To Position Manager")


def position_manager_function():
    print("0: Current Prediction Fetched")
    print("0: Monitoring Current Positions")
    print("0: Assess Improvements To Strategy")
    print("0: Acquire New Positions")
    print("0: Sell old positions")


def safe_system_exit():
    exit_event.set()
    positionThread.join()
    # with sequentialLock:
    sys.exit("System safely shut down")


'Start the program'
if __name__ == '__main__':
    atexit.register(safe_system_exit)
    ready_event = threading.Event()
    position.pass_event(exit_event)
    t = threading.Timer(0.5, prediction.run)
    t.daemon = True
    t.start()
    positionThread = threading.Thread(target=position.run, args=(sequentialLock,))
    positionThread.daemon = True
    positionThread.start()

    while True:
        pass
