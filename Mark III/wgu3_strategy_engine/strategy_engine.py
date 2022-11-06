class StrategyEngine:
    def __init__(self):
        print("Strategy Engine module ..... instantiated")

    def reformat(self):
        print("data reformatted")

    def run(self):
        print("run finished")
        return False


class StrategyEngineBT(StrategyEngine):
    def __init__(self):
        print("Strategy Engine module instantiated --- Backtesting enabled ---")

    def run(self):
        self.reformat()
        print("run back-tested")