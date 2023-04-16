if '__file__' in globals():

    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__)))
    PROJECT_DIR = '/Users/hyewon/Documents/Projects/deep-learning-from-scratch/DeZero'
    sys.path.append(PROJECT_DIR)

import numpy as np
import pandas as pd
import math

from dezero import Variable, Model, optimizers
import dezero.functions as F
from dezero.models import MLP
from dezero.datasets import get_spiral

import matplotlib.pyplot as plt

if __name__ == '__main__':

    np.random.seed(0)

    # 하이퍼파라미터 설정
    max_epoch = 300
    batch_size = 30
    hidden_size = 10
    lr = 1.0

    x, t = get_spiral(train=True)
    VISUALIZE = False
    if VISUALIZE:
        colors = ['pink', 'skyblue', 'lightgreen']
        df = pd.DataFrame({
            'x': x[:, 0].tolist(), 'y': x[:, 1].tolist(), 't': t.tolist()
        })
        for idx, color in zip(range(3), colors):
            df[df['t'] == idx].plot.scatter('x', 'y', marker='o', color=color, ax=plt.gca(), label=idx)
        plt.show()

    model = MLP((hidden_size, 3))
    optimizer = optimizers.SGD(lr)
    optimizer.setup(model)

    data_size = len(x)
    max_iter = math.ceil(data_size / batch_size)

    avg_loss_list = []
    for epoch in range(max_epoch):
        index = np.random.permutation(data_size)
        sum_loss = 0

        for i in range(max_iter):
            batch_index = index[i * batch_size : (i+1) * batch_size]
            batch_x = x[batch_index]
            batch_t = t[batch_index]

            y = model(batch_x)
            loss = F.softmax_cross_entropy(y, batch_t)
            model.cleargrads()
            loss.backward()
            optimizer.update()
            sum_loss += float(loss.data) * len(batch_t)
        
        avg_loss = sum_loss / data_size
        avg_loss_list.append(avg_loss)
        print(f"epoch {epoch+1}, loss {avg_loss:.2f}")
    
    plt.plot(avg_loss_list)
    plt.show()

