"""
Logistic classifier implemented from scratch
By Chenyang Huang, Sept, 2019

Parameter initialization: Uniform[-0.5, 0.5]
"""

import numpy as np


class LogisticClassifier:
    def __init__(self, dim=2000, lr=0.1):
        self.dim = dim
        self.theta = np.random.uniform(-0.5, 0.5, dim)
        self.bias = np.asarray(np.random.uniform(-0.5, 0.5))  # force to generate a np object to call by ref
        self.lr = lr

    def forward(self, data):
        """
        :param data: data.size() = BS * DIM
        :return: BS * 1
        """
        return [1.0 / (1 + np.exp(- np.dot(x, self.theta) - self.bias)) for x in data]

    def __call__(self, data):
        return self.forward(data)

    def parameters(self):
        return {'theta': self.theta,
                'bias': self.bias}

    def gradient_decent_step(self, X, y, y_pred):
        y = np.asarray(y)
        y_pred = np.asarray(y_pred)
        # theta
        gradient = np.zeros(self.theta.shape[0])
        for x, delta in zip(X, y_pred - y):
            gradient += np.dot(x, delta)
        gradient /= len(y)
        self.theta -= self.lr * gradient
        # bias
        gradient = np.mean(y_pred - y)
        self.bias -= self.lr * gradient
