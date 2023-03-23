if '__file__' in globals():

    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__)))
    PROJECT_DIR = '/Users/hyewon/Documents/Projects/deep-learning-from-scratch/DeZero'
    sys.path.append(PROJECT_DIR)

import numpy as np
import pandas as pd
from dezero import Variable, Model
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

    lr = 0.2
    iters = 10000
    hidden_size = 10

    class TwoLayerNet(Model):
        def __init__(self, hidden_size, out_size):
            super().__init__()
            self.l1 = L.Linear(hidden_size)
            self.l2 = L.Linear(out_size)
        
        def forward(self, x):
            y = self.l1(x)
            y = F.sigmoid(y)
            y = self.l2(y)
            return y
    
    model = TwoLayerNet(hidden_size, 1)

    for i in range(iters):
        y_pred = model(x)
        loss = F.mean_squared_error(y, y_pred)

        model.cleargrads()
        loss.backward()

        for p in model.params():
            p.data -= lr * p.grad.data
        if i % 1000 == 0:
            print(loss)

    final_pred = model(x)

    df = pd.DataFrame({
        'x': x.reshape(100,).tolist(), 'y': y.reshape(100,).tolist()
    })
    df.plot.scatter('x', 'y')
    plt.scatter(x, final_pred.data.reshape(100,).tolist(), color='red')
    plt.show()
