if '__file__' in globals():

    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__)))
    PROJECT_DIR = '/Users/hyewon/Documents/Projects/deep-learning-from-scratch/DeZero'
    sys.path.append(PROJECT_DIR)

import numpy as np

from dezero.core_simple import *
from dezero.utils import plot_dot_graph


if __name__ == '__main__':

    x0 = Variable(np.array(0.0))
    x1 = Variable(np.array(2.0))

    lr = 0.001
    iters = 1000

    for i in range(iters):
        print(x0, x1)

        y = rosenbrock(x0, x1)

        x0.cleargrad()
        x1.cleargrad()
        y.backward()

        x0.data -= lr * x0.grad
        x1.data -= lr * x1.grad
