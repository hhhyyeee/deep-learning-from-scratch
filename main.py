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

    x = Variable(np.random.randn(2, 3))
    W = Variable(np.random.randn(3, 4))
    y = F.matmul(x, W)
    y.backward()
    
    print(x.grad.shape)
    print(W.grad.shape)
