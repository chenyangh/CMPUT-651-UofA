

class GradientDecent:
    def __init__(self, paras, lr):
        self.paras = paras
        self.lr = lr

    def step(self, loss):
        for para in self.paras:
            para -= self.lr * loss



