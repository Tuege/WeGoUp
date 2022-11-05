# Mark III

## System Architecture
![image](https://user-images.githubusercontent.com/47752280/200125891-c2d606e8-fffa-482e-a049-a397e995bb70.png)

### Price Prediction:
```python
import wgu3_GAN as prediction
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
import wgu3_profit-risk.py as confidence
```

- calculate all points (long and short) where after comission and safety margin a profit can be made (using now, low and high of period as purchase times?)
- calculate ratio between potential gain and loss for each point
  - potentially take confidence percentage in the direction and value predictions into consideration to calculate probability distribution

### Strategy Engine
```python
import wgu3_strategyEngine as strategy
```

Simple initial strategy for stock:
```
if predicted profit >= x%:
  - place market order and a limit sell order at y% below expected peak
    (y is based ont he mean relative error of the GAN)
  - place safety limit sell order based on the mean relative error of
    the lowest expected price (maybe go lower)
```

Next implement algorithm that takes into account the profit/risk ratio and later other factors that may be relevant when making deisions about investment strategies


### Order/Position Manager:
- shedule oders
- monitor all currentpositions. if any deviate more than the prediction error then initialise Crisis Manager.
    
