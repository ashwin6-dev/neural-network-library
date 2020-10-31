import numpy as np

class MeanSquaredError:

    def get_loss(self, error):
        return error * error
