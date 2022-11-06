class ProfitRisk:
    def __init__(self):
        print("Profit/Risk module ......... instantiated")

    def reformat(self):
        print("data reformatted")

    def run(self):
        print("run finished")
        return False


class ProfitRiskBT(ProfitRisk):
    def __init__(self):
        print("Profit/Risk module instantiated --- Backtesting enabled ---")

    def run(self):
        self.reformat()
        print("run back-tested")