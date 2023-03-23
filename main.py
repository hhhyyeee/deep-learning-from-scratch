if '__file__' in globals():

    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__)))
    PROJECT_DIR = '/Users/hyewon/Documents/Projects/deep-learning-from-scratch/DeZero'
    sys.path.append(PROJECT_DIR)

import numpy as np
import pandas as pd
from dezero import Variable
import dezero.functions as F
import dezero.layers as L
import matplotlib.pyplot as plt

if __name__ == '__main__':

    np.random.seed(0)
    x = np.random.rand(100, 1)
    y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

    # df = pd.DataFrame({
    #     'x': x.reshape(100,).tolist(), 'y': y.reshape(100,).tolist()
    # })
    # plt.figure(figsize=(10, 10))
    # df.plot.scatter('x', 'y')
    # plt.show()

    l1 = L.Linear(10)
    l2 = L.Linear(1)

    def predict(x):
        y = l1(x)
        y = F.sigmoid(y)
        y = l2(y)
        return y
    
    lr = 0.2
    iters = 10000

    for i in range(iters):
        y_pred = predict(x)
        loss = F.mean_squared_error(y, y_pred)

        l1.cleargrads()
        l2.cleargrads()
        loss.backward()

        for l in [l1, l2]:
            for p in l.params():
                p.data -= lr * p.grad.data
        if i % 1000 == 0:
            print(loss)

    final_pred = predict(x)

    df = pd.DataFrame({
        'x': x.reshape(100,).tolist(), 'y': y.reshape(100,).tolist()
    })
    df.plot.scatter('x', 'y')
    plt.scatter(x, final_pred.data.reshape(100,).tolist(), color='red')
    plt.show()
