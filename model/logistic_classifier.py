"""
Logistic classifier implemented from scratch
By Chenyang Huang, Sept, 2019

Parameter initialization: Uniform[-0.5, 0.5]
"""

import numpy as np


class LogisticClassifier:
    def __init__(self, dim=2000):
        self.dim = dim
        self.theta = np.random.uniform(-0.5, 0.5, dim)
        self.bias = np.asarray(np.random.uniform(-0.5, 0.5))  # force to generate a np object to call by ref

    def forward(self, data):
        """
        :param data: data.size() = BS * DIM
        :return: BS * 1
        """
        return [1.0 / (1 + np.exp(-np.dot(x, self.theta) - self.bias)) for x in data]

    def __call__(self, data):
        return self.forward(data)

    def parameters(self):
        return self.theta, self.bias
