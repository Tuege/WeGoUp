
backtesting = True

import pandas as pd
import numpy as np
import yfinance as yf

if backtesting:
    import wgu3_gan.gan_bt           as prediction
    #from wgu3_profit_risk       import profit_risk_bt   as feasibility
    #from wgu3_strategyEngine    import algorithm_bt     as strategy
    prediction = modules.wgu3_gan.gan_bt()
else:
    import wgu3_gan.gan              as prediction
    #from wgu3_profit_risk       import profit_risk      as feasibility
    #from wgu3_strategyEngine    import algorithm        as strategy


print("Hello World!")
print(prediction.run())