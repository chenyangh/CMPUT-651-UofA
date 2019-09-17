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

    def forward(self, data):
        """
        :param data: data.size() = BS * DIM
        :return: BS * 1
        """
        z_list = [self.sigmoid(np.matmul(x, self.W1) + self.b1) for x in data]
        output = [self.sigmoid(np.matmul(z, self.W2) + self.b2) for z in z_list]
        return output

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.e ** -x)

    def __call__(self, data):
        return self.forward(data)

    def gradient_decent_step(self, X, y, y_pred):
        y = np.asarray(y)
        y_pred = np.asarray(y_pred)
        # W2
        old_W2 = deepcopy(self.W2)
        gradient = np.zeros(self.W2.shape[0])
        l1_out = [self.sigmoid(np.matmul(x, self.W1) + self.b1) for x in X]
        for x, delta in zip(l1_out, y_pred.reshape(-1) - y):
            gradient += np.dot(x, delta)
        gradient /= len(y)
        self.W2 -= self.lr * gradient

        # b2
        old_b2 = deepcopy(self.b2)
        gradient = np.mean(y_pred - y)
        self.b2 -= self.lr * gradient

        # W1
        gradient = np.zeros(self.W1.shape[0])
        for x, delta in zip(X, self.W2 - old_W2):
            gradient += np.dot(x, delta)
        self.W1 -= self.lr * gradient

        # b1
        gradient = np.mean(self.b2 - old_b2)
        self.b1 -= self.lr * gradient


