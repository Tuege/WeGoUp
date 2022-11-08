class PositionManager:
    def __init__(self):
        print("Position Manager module .... instantiated")

    def runOpportunity(self):
        print("Opportunity: Finding profitable time periods")

    def run(self):
        self.runOpportunity()


class PositionManagerBT(PositionManager):
    def __init__(self):
        print("Position Manager module instantiated --- Backtesting enabled ---")

