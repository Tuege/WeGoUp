
backtesting = False

import pandas as pd
import numpy as np
import yfinance as yf

import threading
from wgu3_gan import gan as prediction
#import wgu3_gan
#from wgu3_gan.gan import live

if backtesting:
    pass
    #import wgu3_gan.gan_bt      as prediction
    #from wgu3_profit_risk       import profit_risk_bt   as feasibility
    #from wgu3_strategyEngine    import algorithm_bt     as strategy
else:
    pass
    #import wgu3_gan.gan         as prediction
    #from wgu3_profit_risk       import profit_risk      as feasibility
    #from wgu3_strategyEngine    import algorithm        as strategy


print("Hello World!")
p = prediction.live()
print(p.run())

#%%
'New execution section'
