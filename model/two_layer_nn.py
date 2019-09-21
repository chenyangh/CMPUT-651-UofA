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

        cached = {'z1': np.asarray(z1),
                  'p1': np.asarray(p1),
                  'z2': np.asarray(z2),
                  'p2': np.asarray(p2)}
        return np.asarray(p2), cached

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.e ** -x)

    @staticmethod
    def d_sigmoid(x):
        return np.multiply((1 - x), x)

    def __call__(self, data):
        return self.forward(data)

    def gradient_decent_step(self, X_batch, y_batch, y_hat_batch, cached):
        # dJdp2 = - y / p2 + (1 - y) / (1 - p2)
        dJdp2 = - np.divide(y_batch, cached['p2']) + np.divide((1 - y_batch), (1 - cached['p2']))
        dp2dz2 = self.d_sigmoid(cached['z2'])
        dJdz2 = np.multiply(dJdp2, dp2dz2)
        dz2dw2 = cached['p1']
        dJdw2 = np.divide(dJdz2, dz2dw2)  # FIXME
        dJb2 = dJdz2
        dz2dp1 = self.W2
        dJdp1 = np.multiply(dJdz2, dz2dp1)  # FIXME
        dp1dz1 = self.d_sigmoid(cached['z1'])
        dJdz1 = np.multiply(dJdp1, dp1dz1)
        dz1dw1 = X_batch
        dJdw1 = np.multiply(dJdz1, dz1dw1)
        dJb1 = dJdz1

        # update parameters
        self.W1 -= self.lr * dJdw1
        self.b1 -= self.lr * dJb1
        self.W2 -= self.lr * dJdw2
        self.b2 -= self.lr * dJb2







