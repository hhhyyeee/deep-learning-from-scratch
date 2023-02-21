if '__file__' in globals():

    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__)))
    PROJECT_DIR = '/Users/hyewon/Documents/Projects/deep-learning-from-scratch/DeZero'
    sys.path.append(PROJECT_DIR)

import numpy as np

from dezero.core_simple import *
from dezero.utils import plot_dot_graph


if __name__ == '__main__':

    x = Variable(np.array(1.0))
    y = Variable(np.array(1.0))
    z = goldstein(x, y)
    z.backward()

    plot_dot_graph(z)
