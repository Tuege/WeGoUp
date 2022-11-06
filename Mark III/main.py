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
from wgu3_profit_risk import profit_risk
from wgu3_strategy_engine import strategy_engine

#%% Backtesting [enable/disable]
backtesting = True

#%% Module Object Instantiation

if backtesting:
    prediction = gan.GanBT()
    feasibility = profit_risk.ProfitRiskBT()
    strategy = strategy_engine.StrategyEngineBT()
else:
    prediction = gan.Gan()
    feasibility = profit_risk.ProfitRisk()
    strategy = strategy_engine.StrategyEngine()

#%% Threads Setup

stockDataLock = threading.Lock()
technicalIndicatorLock = threading.Lock()
correlatedAssetsLock = threading.Lock()
fourierLock = threading.Lock()
arimaLock = threading.Lock()
predictionsLock = threading.Lock()

stockDataLock.acquire()
stockDataLock.release()

prediction.run(predictionsLock)
feasibility.run()
strategy.run()


#%% System Exit
sys.exit(101)
