class Gan:
    def __init__(self):
        print("Gan module ................. instantiated")

    def reformat(self):
        print("data reformatted")

    def run(self, lock):
        print("run finished")
        return False

    def updatePrediction(self, output):
        publicPrediction = output

    def runGan(self):
        print("Prediction: Predicting Stock Movement\n")


class GanBT(Gan):
    def __init__(self):
        print("Gan module instantiated --- Backtesting enabled ---")

    def run(self, lock):
        self.runGan()
        while not lock.acquire(blocking=True):
            print("Prediction: Block Failed!")
        self.updatePrediction(1)
        print("Prediction: Prediction updated\n")
        lock.release()
