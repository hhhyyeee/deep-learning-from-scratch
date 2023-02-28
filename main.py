if '__file__' in globals():

    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__)))
    PROJECT_DIR = '/Users/hyewon/Documents/Projects/deep-learning-from-scratch/DeZero'
    sys.path.append(PROJECT_DIR)

import numpy as np

from dezero.core_simple import *
from dezero.utils import plot_dot_graph


if __name__ == '__main__':

    x = Variable(np.array(np.pi / 4))
    y = my_sin(x)
    y.backward()

    print(y.data)
    print(x.grad)
