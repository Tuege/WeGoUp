# %% Backtesting
backtesting = False

# %% Imports

import pandas as pd
import numpy as np
import yfinance as yf

import threading

try:
    del sys.modules["wgu3_gan.gan"]
    del wgu3_gan.gan
except (KeyError, NameError):
    pass
try:
    print(str(wgu3_profit_risk.profit_risk))
    del sys.modules["wgu3_profit_risk.profit_risk"]
    del wgu3_profit_risk.profit_risk
except (KeyError, NameError):
    pass
try:
    del sys.modules["wgu3_strategy_engine.strategy_engine"]
    del wgu3_strategy_engine.strategy_engine
except (KeyError, NameError):
    pass

from wgu3_gan import gan
from wgu3_profit_risk import profit_risk
from wgu3_strategy_engine import strategy_engine

if backtesting:
    prediction = gan.GanBT()
    feasibility = profit_risk.ProfitRiskBT()
    strategy = strategy_engine.StrategyEngineBT()
else:
    prediction = gan.Gan()
    feasibility = profit_risk.ProfitRisk()
    strategy = strategy_engine.StrategyEngine()

# %%

print("Hello World!")
prediction.run()
# p = gan.backtesting()
# print(p.run())


# %%
