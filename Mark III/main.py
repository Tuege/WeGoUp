"""
Tuege Neumann
06.11.2022

A GAN based approach to Algorithmic Trading

"""
#%% Imports
import sys
import threading

import pandas as pd
import numpy as np
import yfinance as yf

from wgu3_gan import gan
from wgu3_strategy_engine import strategy_engine
from wgu3_position_manager import position_manager


#%% Backtesting [enable/disable]
backtesting = True

#%% Module Object Instantiation

if backtesting:
    prediction = gan.GanBT()
    strategy = strategy_engine.StrategyEngineBT()
    position = position_manager.PositionManagerBT()
else:
    prediction = gan.Gan()
    strategy = strategy_engine.StrategyEngine()
    position = position_manager.PositionManager()

print("____________________________________________________________________\n")

#%% Threads Setup

'Locks'
stockDataLock = threading.Lock()
technicalIndicatorLock = threading.Lock()
correlatedAssetsLock = threading.Lock()
fourierLock = threading.Lock()
arimaLock = threading.Lock()
predictionLock = threading.Lock()

'Threads'
predictionThread = threading.Thread(target=prediction.run, args=(predictionLock,))
strategyThread = threading.Thread(target=strategy.run, args=(predictionLock,))
positionThread = threading.Thread(target=position.run, args=(predictionLock,))

'Start the program'
predictionThread.start()
strategyThread.start()
positionThread.start()

#%% System Exit
sys.exit(101)
