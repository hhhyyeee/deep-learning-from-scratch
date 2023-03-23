if '__file__' in globals():

    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__)))
    PROJECT_DIR = '/Users/hyewon/Documents/Projects/deep-learning-from-scratch/DeZero'
    sys.path.append(PROJECT_DIR)

import numpy as np
import pandas as pd
from dezero import Variable
from dezero.core import exp
import dezero.functions as F
from dezero.utils import plot_dot_graph
import matplotlib.pyplot as plt

if __name__ == '__main__':

    np.random.seed(0)
    x = np.random.rand(100, 1)
    y = 5 + 2 * x + np.random.rand(100, 1)

    # df = pd.DataFrame({
    #     'x': x.reshape(100,).tolist(), 'y': y.reshape(100,).tolist()
    # })
    # plt.figure(figsize=(10, 10))
    # df.plot.scatter('x', 'y')
    # plt.show()

    W = Variable(np.zeros((1, 1)))
    b = Variable(np.zeros(1))

    def predict(x):
        y = F.matmul(x, W) + b
        return y
    
    lr = 0.1
    iters = 100

    for i in range(iters):
        y_pred = predict(x)
        loss = F.mean_squared_error_simple(y, y_pred)

        W.cleargrad()
        b.cleargrad()
        loss.backward()

        W.data -= lr * W.grad.data
        b.data -= lr * b.grad.data
        print(W, b, loss)
