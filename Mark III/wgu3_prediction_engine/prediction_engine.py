import sys
import threading
from gan import gan

class PredictionEngine:
    def __init__(self):
        print("Gan module ................. instantiated")

    def reformat(self):
        print("data reformatted")

    def run(self):
        with priority_lock:
            pass
        t = threading.Timer(0.5, self.run)
        t.daemon = True
        t.start()
        print("1: Strategy recalculating")

    def updatePrediction(self, output):
        publicPrediction = output

    def runGan(self):
        print("Prediction: Predicting Stock Movement")


class PredictionEngineBT(PredictionEngine):
    def __init__(self):
        print("Gan module instantiated              --- Backtesting enabled ---")


class _RandomClass:
    def __init__(self):
        pass

    def _run(self):
        print("This class is not protected")
