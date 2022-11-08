class Gan:
    def __init__(self):
        print("Gan module ................. instantiated")

    def reformat(self):
        print("data reformatted")

    def run(self, lock):
        self.runGan()
        with lock:
            self.updatePrediction(1)
            print("Prediction: Prediction updated")
        print("run finished")

    def updatePrediction(self, output):
        publicPrediction = output

    def runGan(self):
        print("Prediction: Predicting Stock Movement")


class GanBT(Gan):
    def __init__(self):
        print("Gan module instantiated              --- Backtesting enabled ---")

