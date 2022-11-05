
backtesting = True

import pandas as pd
import numpy as np
import yfinance as yf

import threading

if backtesting:
    import wgu3_gan.gan_bt      as prediction
    #from wgu3_profit_risk       import profit_risk_bt   as feasibility
    #from wgu3_strategyEngine    import algorithm_bt     as strategy
else:
    import wgu3_gan.gan         as prediction
    #from wgu3_profit_risk       import profit_risk      as feasibility
    #from wgu3_strategyEngine    import algorithm        as strategy


print("Hello World!")
print(prediction.run())

#%%
