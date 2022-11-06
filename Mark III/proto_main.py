
backtesting = False

import pandas as pd
import numpy as np
import yfinance as yf

import threading
del sys.modules["wgu3_gan.gan"]
del wgu3_gan.gan
from wgu3_gan import gan
#del sys.modules["moduleName"]
#del moduleName
#import wgu3_gan.gan
#from wgu3_gan import gan
#import wgu3_gan
#from wgu3_gan.gan import live

#print(dir(wgu3_gan.gan))


if backtesting:
    pass
    #import wgu3_gan.gan_bt      as prediction
    prediction = gan.GanBT()
    #from wgu3_profit_risk       import profit_risk_bt   as feasibility
    #from wgu3_strategyEngine    import algorithm_bt     as strategy
else:
    pass
    #import wgu3_gan.gan         as prediction
    prediction = gan.Gan()
    #from wgu3_profit_risk       import profit_risk      as feasibility
    #from wgu3_strategyEngine    import algorithm        as strategy


print("Hello World!")
prediction.run()
#p = gan.backtesting()
#print(p.run())

#%%
'New execution section'

