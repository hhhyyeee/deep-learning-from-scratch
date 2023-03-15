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

    x = Variable(np.array(1.0))
    y = F.tanh(x)
    x.name = 'x'
    y.name = 'y'
    y.backward(create_graph=True)

    iters = 5
    for i in range(iters):
        gx = x.grad
        x.cleargrad()
        gx.backward(create_graph=True)
    
    gx = x.grad
    gx.name = f"gx{str(iters+1)}"
    plot_dot_graph(gx, verbose=False, to_file='tanh.png')

