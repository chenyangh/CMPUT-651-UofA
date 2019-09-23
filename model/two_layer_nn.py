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
        z1 = np.matmul(X, self.W1) + self.b1
        p1 = self.sigmoid(z1)
        z2 = np.matmul(p1, self.W2) + self.b2
        p2 = self.sigmoid(z2)

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
        return np.multiply((1 - x), x)

    def __call__(self, data):
        return self.forward(data)

    def gradient_decent_step(self, X_batch, y_batch, cached):
        bs = X_batch.shape[0]  # batch size
        # dJdp2 = - y / p2 + (1 - y) / (1 - p2)
        dJdp2 = - np.divide(y_batch, cached['p2']) + np.divide((1 - y_batch), (1 - cached['p2']))
        dp2dz2 = self.d_sigmoid(cached['z2'])
        dJdz2 = np.dot(dJdp2, dp2dz2)
        dz2dw2 = cached['p1']
        # dJdw2 = dJdz2 * dz2dw2
        dJdw2 = np.matmul(np.tile(dJdz2, (1, bs)), dz2dw2).reshape(-1)
        dJb2 = dJdz2
        dz2dp1 = self.W2
        dJdp1 = np.dot(dJdz2, dz2dp1)
        dp1dz1 = self.d_sigmoid(cached['z1'])
        dJdz1 = np.sum(np.asarray([np.multiply(dJdp1, dp1dz1_i) for dp1dz1_i in dp1dz1]), axis=0)
        dz1dw1 = X_batch
        dJdw1 = np.matmul(np.tile(dJdz1, (bs, 1)).T, dz1dw1).T
        dJb1 = dJdz1

        # update parameters
        self.W2 -= self.lr * dJdw2 / bs
        self.b2 -= self.lr * dJb2 / bs
        self.W1 -= self.lr * dJdw1 / bs
        self.b1 -= self.lr * dJb1 / bs








