class StrategyEngine:
    def __init__(self):
        print("Strategy Engine module ..... instantiated")

    def method_a(self):
        pass

    def method_b(self):
        pass

    def run(self, prediction_thread, lock):
        print("run finished")
        return False


class StrategyEngineBT(StrategyEngine):
    def __init__(self):
        print("Strategy Engine module instantiated --- Backtesting enabled ---")

    def run(self, prediction_thread, lock):
        prediction_thread.start()
        while not lock.acquire(blocking=True):
            pass
        print("Strategy: reading out the prediction results\n")
        lock.release()
        print("Strategy: run back-tested")
