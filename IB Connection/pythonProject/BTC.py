from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract

class MyWrapper(EWrapper):
    def tickPrice(self, reqId, tickType, price, attrib):
        if tickType == 66: # tickType 66 corresponds to the last trade price
            print("Current price of Bitcoin futures: ${:.2f}".format(price))

class MyClient(EClient):
    def __init__(self, wrapper):
        EClient.__init__(self, wrapper)

wrapper = MyWrapper()
client = MyClient(wrapper)

client.connect("127.0.0.1", 7497, clientId=0)

contract = Contract()
contract.symbol = "BTC"
contract.secType = "CRYPTO"
contract.exchange = "PAXOS"
contract.currency = "USD"

client.reqMarketDataType(1)
client.reqMktData(1, contract, "", False, False, [])

client.run()

client.disconnect()