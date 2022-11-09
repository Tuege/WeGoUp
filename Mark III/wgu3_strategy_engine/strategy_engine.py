class StrategyEngine:
    def __init__(self):
        print("Strategy Engine module ..... instantiated")

    def opportunity(self):
        print("Opportunity: Finding profitable time periods")

    def feasibility(self):
        pass

    def get_prediction(self, prediction_lock):
        with prediction_lock:
            #TODO: add the correct variable for prediction
            self.current_prediction = None
            print("Strategy Engine: reading out the prediction results")

    def run(self, prediction_thread, lock):
        while True:
            self.get_prediction(lock)
            print("Strategy Engine: Strategising about")


class StrategyEngineBT(StrategyEngine):
    def __init__(self):
        print("Strategy Engine module instantiated  --- Backtesting enabled ---")
