"""
Simple Cross entropy loss implementation by numpy
By Chenyang, Sept, 2019
"""

import numpy as np


class CrossEntropyLoss:
    def __init__(self, reduce=True):
        self.reduce = reduce

    def __call__(self, y_gold, y_pred):

        loss = - np.sum([y_i * np.log(y_hat_i) + (1 - y_i) * np.log(1 - y_hat_i)
                         for y_i, y_hat_i in zip(y_gold, y_pred)])

        if self.reduce:
            loss /= len(y_pred)
        return loss


