"""
Simple Cross entropy loss implementation by numpy
By Chenyang, Sept, 2019
"""

import numpy as np


class CrossEntropyLoss:
    @staticmethod
    def single_ce(gold, pred):
        if gold == 1:
            return - np.log(pred)
        else:
            return - np.log(1 - pred)

    def __call__(self, y_gold, y_pred):
        loss = 0
        for gold, pred in zip(y_gold, y_pred):
            loss += self.single_ce(gold, pred)
        return loss


