if '__file__' in globals():

    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__)))
    PROJECT_DIR = '/Users/hyewon/Documents/Projects/deep-learning-from-scratch/DeZero'
    sys.path.append(PROJECT_DIR)

import numpy as np
import pandas as pd
import math

from dezero import Variable, Model, optimizers, no_grad
import dezero.functions as F
from dezero.models import MLP
from dezero.datasets import Spiral
from dezero.dataloaders import DataLoader

import matplotlib.pyplot as plt

if __name__ == '__main__':

    np.random.seed(0)

    # 하이퍼파라미터 설정
    max_epoch = 300
    batch_size = 30
    hidden_size = 10
    lr = 1.0

    train_set = Spiral(train=True)
    test_set = Spiral(train=False)
    train_loader = DataLoader(train_set, batch_size)
    test_loader = DataLoader(test_set, batch_size, shuffle=False)

    model = MLP((hidden_size, 3))
    optimizer = optimizers.SGD(lr)
    optimizer.setup(model)

    data_size = len(train_set)
    max_iter = math.ceil(data_size / batch_size)

    avg_loss_list, avg_acc_list = [], []
    for epoch in range(max_epoch):
        index = np.random.permutation(data_size)

        sum_loss, sum_acc = 0, 0
        for i in range(max_iter):
            for x, t in train_loader:
                x, t = Variable(x), Variable(t)

                y = model(x)
                loss = F.softmax_cross_entropy(y, t)
                acc = F.accuracy(y, t)
                model.cleargrads()
                loss.backward()
                optimizer.update()
                sum_loss += float(loss.data) * len(t)
                sum_acc += float(acc) * len(t)
        
        avg_loss = sum_loss / data_size
        avg_acc = sum_acc / data_size

        print(f"epoch {epoch+1}, loss {avg_loss:.2f}, accuracy {avg_acc:.2f}")

        sum_loss, sum_acc = 0, 0
        with no_grad():
            for x, t in test_loader:
                x, t = Variable(x), Variable(t)
                y = model(x)
                loss = F.softmax_cross_entropy(y, t)
                acc = F.accuracy(y, t)
                sum_loss += float(loss.data) * len(t)
                sum_acc += float(acc) * len(t)

            avg_loss = sum_loss / len(test_set)
            avg_acc = sum_acc / len(test_set)
            avg_loss_list.append(avg_loss)
            avg_acc_list.append(avg_acc)
            print(f"epoch {epoch+1}, test loss {avg_loss:.2f}, test accuracy {avg_acc:.2f}")

    plt.figure(figsize=(10, 5))
    plt.plot(avg_loss_list)
    plt.plot(avg_acc_list)
    plt.show()

