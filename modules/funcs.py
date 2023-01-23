import numpy as np

from modules.utils import *


def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


# -----
def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)

def add(x0, x1):
    return Add()(x0, x1)

def multiply(x0, x1):
    return Multiply()(x0, x1)

def divide(numer, denom):
    return Divide()(numer, denom)
# -----


# -----
class Square(Function):
    def forward(self, x):
        return x ** 2
    
    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx

class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y
    
    def backward(self, gy):
        return gy, gy

class Multiply(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y
    
    def backward(self, gys):
        pass

class Divide(Function):
    def forward(self, numer, denom):
        if denom.data == 0:
            raise ZeroDivisionError
        y = numer / denom
        return y

