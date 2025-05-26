import math
from math import exp
prob = []
max_sofar = None
sum = 0.0
class OnlineSoftmax:
    def __init__(self):
        self.max_sofar = None
        self.sum = 0.0
        self.prob = []
    def online_softmax(self, x):
        if self.max_sofar is None:
            self.max_sofar = x
            self.prob.append(1)
            self.sum = 1
            return self.prob      
        dif = max(0, x - self.max_sofar)  
        old_sum = self.sum
        self.max_sofar = max(x, self.max_sofar) 
        self.sum /= exp(dif);
        self.sum += exp(x - self.max_sofar)
        scale_factor = old_sum / (exp(dif) * self.sum)
        for i, _ in enumerate(self.prob):
            self.prob[i] *= scale_factor
        self.prob.append(exp(x - self.max_sofar) / self.sum)
        return self.prob
a = 1;
b = 1;
c = 1;
d = 1;
online = OnlineSoftmax()
print(online.online_softmax(a))
print(online.online_softmax(b))
print(online.online_softmax(c))
print(online.online_softmax(d))
