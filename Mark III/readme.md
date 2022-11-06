# Mark III

## System Architecture
![image](https://user-images.githubusercontent.com/47752280/200125891-c2d606e8-fffa-482e-a049-a397e995bb70.png)

### Price Prediction
```python
from wgu3_gan import gan

# Live trading .... gan.Gan()
# Backtesting ..... gan.GanBT()
prediction = gan.Gan()
```

GAN model taking the following inputs:
- [ ] ARIMA model
- [ ] price
- [ ] volume
- [ ] technical indicators
- [ ] correlated assets
- [ ] fourier analysis
- [ ] sentiment analysis
- [ ] autoencoders

### Profit/Risk Calculations
```python
import wgu3_profit-risk as feasability
```

- calculate all points (long and short) where after comission and safety margin a profit can be made (using now, low and high of period as purchase times?)
- calculate ratio between potential gain and loss for each point
  - potentially take confidence percentage in the direction and value predictions into consideration to calculate probability distribution

### Strategy Engine
```python
import wgu3_strategyEngine as strategy
```

Evaluate all possible orders using confidence values, upside/downside potential and potentially later an Order Evalutaion CNN.

Simple initial strategy for stock:
```
if predicted profit >= x%:
  - place market order and a limit sell order at y% below expected peak
    (y is based ont he mean relative error of the GAN)
  - place safety limit sell order based on the mean relative error of
    the lowest expected price (maybe go lower)
```

Next implement algorithm that takes into account the profit/risk ratio and later other factors that may be relevant when making deisions about investment strategies


### Realtime Position/Exposure Manager
```python
import wgu3_orderBook as orderBook
```

- schedule oders
- monitor all currentpositions. if any deviate more than the prediction error then initialise Crisis Manager.

__The following are some popular risk management rules:__

- [ ] Position limit: Control the upper limit of the position of a specified instrument, or the sum of all positions of instruments for a specified product.
- [ ] Single-order limit: Control the upper limit of the volume of single order. Sometimes, control the lower limit of the volume of single order, which means that the quantity of your order must be a multiple of it.
- [ ] Money control: Control the margin of all positions not to exceed the balance of the account.
- [ ] Illegal price detection: Ensure the price is within a reasonable range, such as not exceed price limit, or not too far from the current price.
- [ ] Self-trading detection: Ensure the orders from different strategies will not cause any possibility of trade between them.
- [ ] Order cancellation rate: Calculate the order cancellation situation and ensure it does not exceed the limitation of exchange.

### Crisis Manager
```python
import wgu3_crisisManager as crisis
```
    
