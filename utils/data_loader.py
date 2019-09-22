"""
Simple data loader,
By Chenyang Sept, 2019
"""
import numpy as np


class DataLoader:
    def __init__(self, X, y, bs):
        self.X = X
        self.y = y
        self.bs = bs
        self.cur_idx = 0
        self.idx_list = list(range(len(X)))

    def __len__(self):
        return len(self.X)

    def __iter__(self):
        return self

    def __next__(self):
        if self.cur_idx is None:
            self.cur_idx = 0
            raise StopIteration
        else:
            return self._next_batch()

    def _next_index(self):
        if self.cur_idx + self.bs < self.__len__():
            next_index_list = self.idx_list[self.cur_idx: self.cur_idx + self.bs]
            self.cur_idx += self.bs
        else:
            next_index_list = self.idx_list[self.cur_idx:]
            self.cur_idx = None  # indicate epoch end
        return next_index_list

    def _next_batch(self):
        next_index_list = self._next_index()
        return np.asarray([self.X[i] for i in next_index_list]), \
               np.asarray([self.y[i] for i in next_index_list])
