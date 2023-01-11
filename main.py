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

x = Variable(np.array(0.5))
print(x.data)
print(numerical_diff(f, x))

