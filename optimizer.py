from torch import optim


class MyOptimizer():
    def __init__(self, parameters, learning_rate, weight_decay=0) -> None:
        self.params = parameters
        self.lr = learning_rate
        self.weight_decay = weight_decay
    
    def curr_optim(self):
        return self.SGD()
        return self.Adam()
    
    def SGD(self):
        return optim.SGD(params=self.params, lr=self.lr, weight_decay=self.weight_decay)
    
    def Adam(self):
        return optim.Adam(params=self.params, lr=self.lr, weight_decay=self.weight_decay)
