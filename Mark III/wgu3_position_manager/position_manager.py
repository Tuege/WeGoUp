class PositionManager:
    def __init__(self):
        print("Position Manager module .... instantiated")
        self.current_prediction = None
        self.current_strategy_list = None

    def get_prediction(self, prediction_lock):
        with prediction_lock:
            #TODO: add the correct variable for prediction
            self.current_prediction = None
            print("Position Manager: Prediction Data successfully retrieved")

    def get_strategy_list(self, strategy_lock):
        with strategy_lock:
            # TODO: add the correct variable for strategy list
            current_strategy_list = None
            print("Position Manager: Strategy List successfully retrieved")

    def run(self, prediction_lock):
        while True:
            self.get_prediction(prediction_lock)
            print("Position Manager: Monitoring Open Positions")


class PositionManagerBT(PositionManager):
    def __init__(self):
        print("Position Manager module instantiated --- Backtesting enabled ---")

