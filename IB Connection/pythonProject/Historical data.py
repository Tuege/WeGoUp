from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract

import threading
import time


class IBapi(EWrapper, EClient):
	def __init__(self):
		EClient.__init__(self, self)
		self.data = [] #Initialize variable to store candle

	def historicalData(self, reqId, bar):
		print(f'Time: {bar.date} Close: {bar.close}')
		self.data.append([bar.date, bar.close])


def run_loop():
    app.run()


app = IBapi()
app.connect('127.0.0.1', 7497, 123)

# Start the socket in a thread
api_thread = threading.Thread(target=run_loop, daemon=True)
api_thread.start()

time.sleep(2)  # Sleep interval to allow time for connection to server

# Create contract object
ES_futures__contract = Contract()
ES_futures__contract.symbol = 'ES'
ES_futures__contract.secType = 'FUT'
ES_futures__contract.exchange = 'GLOBEX'
ES_futures__contract.lastTradeDateOrContractMonth  = '20221216'

# Request historical candles
app.reqHistoricalData(1, ES_futures__contract, '', '2 D', '1 hour', 'BID', 0, 2, False, [])

time.sleep(5)  # sleep to allow enough time for data to be returned

#Working with Pandas DataFrames
import pandas

df = pandas.DataFrame(app.data, columns=['DateTime', 'Close'])
df['DateTime'] = pandas.to_datetime(df['DateTime'],unit='s')
df.to_csv('ES_futures_hourly.csv')
print('complete')
print(df)


app.disconnect()