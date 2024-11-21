
class PlaneWaveDof:
    def __init__(self, xmin: float, xmax: float, n: int):
        if xmin > xmax:
            raise Exception("Range should be positive")
