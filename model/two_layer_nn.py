"""
Logistic classifier implemented from scratch
By Chenyang Huang, Sept, 2019

Parameter initialization: Uniform[-0.5, 0.5]
"""

import numpy as np
from copy import deepcopy


class TwoLayerNN:
    def __init__(self, input_dim=2000, lr=0.1, hidden_dim=200):
        self.input_dim = input_dim
        self.W1 = np.random.uniform(-0.5, 0.5, (input_dim, hidden_dim))
        self.b1 = np.random.uniform(-0.5, 0.5, hidden_dim)  # force to generate a np object to call by ref
        self.W2 = np.random.uniform(-0.5, 0.5, hidden_dim)
        self.b2 = np.asarray(np.random.uniform(-0.5, 0.5))
        self.lr = lr

    def forward(self, X):
        """
        :param data: data.size() = BS * DIM
        :return: BS * 1
        """
        # z1 = W1 * X + b1
        # p1 = sigmoid(z1)
        # z2 = W2 * p1 + b2
        # p2 = sigmoid(z2)
        z1 = [np.matmul(x, self.W1) + self.b1 for x in X]
        p1 = [self.sigmoid(z) for z in z1]
        z2 = [np.matmul(p, self.W2) + self.b2 for p in p1]
        p2 = [self.sigmoid(z) for z in z2]

        cached = {'z1': z1,
                  'p1': p1,
                  'z2': z2,
                  'p2': p2}
        return p2, cached

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.e ** -x)

    @staticmethod
    def d_sigmoid(x):
        return np.dot((1 - x), x)

    def __call__(self, data):
        return self.forward(data)

    def gradient_decent_step(self, X, y, y_pred, cached):
        y = np.asarray(y)
        y_pred = np.asarray(y_pred)



