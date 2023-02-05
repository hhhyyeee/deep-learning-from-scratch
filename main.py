import sys
import numpy as np

PROJECT_DIR = '/Users/hyewon/Documents/Projects/deep-learning-from-scratch/DeZero'
sys.path.append(PROJECT_DIR)

from modules.funcs import *
from modules.utils import *
from modules.test import *


def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)


if __name__ == '__main__':

    x = Variable(np.array(2.0))
    y = x + np.array(2.0)
    print(y.data)
    y.backward()

    print(y.data)
    print(x.grad)

