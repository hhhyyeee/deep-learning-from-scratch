if '__file__' in globals():

    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__)))
    PROJECT_DIR = '/Users/hyewon/Documents/Projects/deep-learning-from-scratch/DeZero'
    sys.path.append(PROJECT_DIR)

import numpy as np
import pandas as pd

from dezero import Variable, Model, optimizers
import dezero.functions as F
from dezero.models import MLP

import matplotlib.pyplot as plt

if __name__ == '__main__':

    np.random.seed(0)
    x = np.array([[0.2, -0.4], [0.3, 0.5], [1.3, -3.2], [2.1, 0.3]])
    t = np.array([2, 0, 1, 0])

    lr = 0.2
    max_iter = 10000
    hidden_size = 10

    model = MLP((hidden_size, 3))
    y = model(x)
    loss = F.softmax_cross_entropy_simple(y, t)
    print(loss)

    # optimizer = optimizers.SGD(lr)
    # optimizer.setup(model)

    # for i in range(max_iter):
    #     y_pred = model(x)
    #     loss = F.mean_squared_error(y, y_pred)

    #     model.cleargrads()
    #     loss.backward()

    #     optimizer.update()

    #     if i % 1000 == 0:
    #         print(loss)

    # final_pred = model(x)

    # df = pd.DataFrame({
    #     'x': x.reshape(100,).tolist(), 'y': y.reshape(100,).tolist()
    # })
    # df.plot.scatter('x', 'y')
    # plt.scatter(x, final_pred.data.reshape(100,).tolist(), color='red')
    # plt.show()
