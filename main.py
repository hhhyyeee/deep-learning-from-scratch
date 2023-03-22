if '__file__' in globals():

    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__)))
    PROJECT_DIR = '/Users/hyewon/Documents/Projects/deep-learning-from-scratch/DeZero'
    sys.path.append(PROJECT_DIR)

import numpy as np
from dezero import Variable
from dezero.core import exp
import dezero.functions as F
from dezero.utils import plot_dot_graph
import matplotlib.pyplot as plt

if __name__ == '__main__':

    x = np.array([[1, 2, 3], [4, 5, 6]])
    y = F.sum_to(x, (1, 3))
    print(y)

    y = F.sum_to(x, (2, 1))
    print(y)
