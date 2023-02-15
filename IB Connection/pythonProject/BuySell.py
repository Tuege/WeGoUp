from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order

class IBAPI(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.order_id = None

    def nextValidId(self, orderId: int):
        self.order_id = orderId

    def orderStatus(self, orderId, status, filled, remaining, avgFillPrice, permId,
                    parentId, lastFillPrice, clientId, whyHeld, mktCapPrice):
        if self.order_id == orderId:
            print(f"Order {orderId} status: {status}")
            if status == "Filled":
                print(f"Order filled. Price: {avgFillPrice}")
                self.disconnect()

def buy_stock(symbol, exchange, quantity, price):
    ib = IBAPI()
    ib.connect("127.0.0.1", 7497, 0)

    contract = Contract()
    contract.symbol = symbol
    contract.exchange = exchange
    contract.secType = "STK"
    contract.currency = "USD"

    order = Order()
    order.action = "BUY"
    order.orderType = "LMT"
    order.totalQuantity = quantity
    order.lmtPrice = price

    ib.placeOrder(ib.order_id, contract, order)
    ib.run()

def sell_stock(symbol, exchange, quantity, price):
    ib = IBAPI()
    ib.connect("127.0.0.1", 7497, 0)

    contract = Contract()
    contract.symbol = symbol
    contract.exchange = exchange
    contract.secType = "STK"
    contract.currency = "USD"

    order = Order()
    order.action = "SELL"
    order.orderType = "LMT"
    order.totalQuantity = quantity
    order.lmtPrice = price

    ib.placeOrder(ib.order_id, contract, order)
    ib.run()

# Example usage
symbol = "AAPL"
exchange = "SMART"
quantity = 100
price = 200.0

# Buy 100 shares of AAPL stock at $200.0 each
buy_stock(symbol, exchange, quantity, price)

# Sell 100 shares of AAPL stock at $200.0 each
sell_stock(symbol, exchange, quantity, price)