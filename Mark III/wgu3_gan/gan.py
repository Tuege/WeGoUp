class Gan:
    def __init__(self):
        print("Gan module instantiated")

    def reformat(self):
        print("data reformatted")

    def run(self):
        print("run finished")
        return False


class GanBT(Gan):
    def __init__(self):
        print("Gan module instantiated --- Backtesting enabled ---")

    def run(self):
        self.reformat()
        print("run back-tested")
