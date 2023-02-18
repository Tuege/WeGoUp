from ib_insync import *
util.startLoop()

ib = IB()
ib.connect(host='127.0.0.1', port=7497, clientId=0)

eur_usd_contract = Forex('EURUSD', 'IDEALPRO')
ib.qualifyContracts(eur_usd_contract)
eur_usd_data = ib.reqMktData(eur_usd_contract)
eur_usd_data.marketPrice()
print(eur_usd_data.marketPrice())