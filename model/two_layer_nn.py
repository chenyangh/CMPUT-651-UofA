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
        self.hidden_dim = hidden_dim
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

    def d_sigmoid(self, x):
        return np.multiply((1 - self.sigmoid(x)), self.sigmoid(x))

    def __call__(self, data):
        return self.forward(data)

    def gradient_decent_step(self, X, y, cached):
        # X: bs * input_dim
        # y: bs
        bs = X.shape[0]  # get batch size
        # following the notation in forward propagation, y_hat is p2
        # Back-propagate output layer
        # dJdp2 = - y / p2 + (1 - y) / (1 - p2)
        dJdp2 = - np.divide(y, cached['p2']) + np.divide((1 - y), (1 - cached['p2']))  # bs * output_dim
        dp2dz2 = self.d_sigmoid(cached['z2'])  # bs * output_dim
        dJdz2 = np.multiply(dJdp2, dp2dz2)  # bs * output_dim
        dz2dw2 = cached['p1']  # bs * hidden_dim
        # dJdw2 = dJdz2 * dz2dw2
        # dJdw2 = np.multiply(np.tile(dJdz2, (200, 1)).T, dz2dw2)  # bs * hidden_dim * output_dim
        dJdw2 = np.matmul(dz2dw2.T, dJdz2)  # bs * hidden_dim * output_dim
        dJdb2 = dJdz2  # bs * output_dim
        # Back-propagate hiddent layer
        dz2dp1 = self.W2  # bs * hidden_dim
        dJdp1 = dJdz2.reshape(-1, 1) * np.tile(dz2dp1, (bs, 1))  # bs * hidden_dim
        dp1dz1 = self.d_sigmoid(cached['z1'])  # bs * hidden_dim
        dJdz1 = np.multiply(dJdp1, dp1dz1)  # bs * hidden_dim
        dz1dw1 = X  # bs * input_dim
        dJdw1 = np.matmul(dz1dw1.T, dJdz1)  # bs * input_dim * hidden_dim
        dJdb1 = dJdz1  # bs * hidden_dim

        # update parameters
        self.W2 -= self.lr * dJdw2 / bs
        self.b2 -= self.lr * np.mean(dJdb2, axis=0)
        self.W1 -= self.lr * dJdw1 / bs
        self.b1 -= self.lr * np.mean(dJdb1, axis=0)









