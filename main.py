import sys
import numpy as np

PROJECT_DIR = '/Users/hyewon/Documents/Projects/deep-learning-from-scratch/DeZero'
sys.path.append(PROJECT_DIR)

from modules.diff import *
from modules.utils import *


def f(x):
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))


if __name__ == '__main__':

    A = Square()
    B = Exp()
    C = Square()

    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)

    y.grad = np.array(1.0)
    y.backward()
    print(x.grad)

