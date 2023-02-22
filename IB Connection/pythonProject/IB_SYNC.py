from ib_insync import *
import time

util.startLoop()

ib = IB()
ib.connect(host='127.0.0.1', port=7497, clientId=0)


def order_status(trade):
    if trade.orderStatus.status == 'Filled':
        fill = trade.fills[-1]

        print(
            f'{fill.time} - {fill.execution.side} {fill.contract.symbol} {fill.execution.shares} @ {fill.execution.avgPrice}')


# EURUSD
eur_usd_contract = Forex('EURUSD', 'IDEALPRO')
ib.qualifyContracts(eur_usd_contract)
eur_usd_data = ib.reqMktData(eur_usd_contract)

# SPY
spy_contract = Stock('SPY', 'SMART', 'USD')
ib.qualifyContracts(spy_contract)
spy_data = ib.reqMktData(spy_contract)

# SMA
historical_data_spy = ib.reqHistoricalData(
    spy_contract,
    '',
    barSizeSetting='15 mins',
    durationStr='2 D',
    whatToShow='MIDPOINT',
    useRTH=True
)

type(historical_data_spy)
historical_data_spy[-1]
historical_data_spy[-1].open
util.df(historical_data_spy)
# print(util.df(historical_data_spy))

spy_df = util.df(historical_data_spy)
spy_df.close.rolling(20).mean()
spy_df['20SMA'] = spy_df.close.rolling(20).mean()

# print (spy_df)

"""while True:
    ib.sleep(1)
    print("EURUSD: ", eur_usd_data.marketPrice())
    print("SPY: ", spy_data.marketPrice())"""

"""spy_order = MarketOrder("BUY", 100)
trade = ib.placeOrder(spy_contract, spy_order)
trade.log
trade.orderStatus.status

for _ in range(100):
    if not trade.isActive():
        print(f'Your order status - {trade.orderStatus.status}')
        break
    time.sleep(0.5)
else:
    print('Order is still active')

trade.filledEvent += order_status
print(order_status(trade))"""

spy_bracket_order = ib.bracketOrder(
    "BUY",
    100,
    limitPrice=399.6,
    takeProfitPrice=399.7,
    stopLossPrice=399.5
)

for ordr in spy_bracket_order:
    ib.placeOrder(eur_usd_contract, ordr)

spy_bracket_order = ib.bracketOrder('BUY', 100, 399.6, 399.7, 399.5)
