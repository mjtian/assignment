from __future__ import division
import torch
from torch import optim
import torch.nn as nn
import numpy as np

from utils import load_MNIST, random_draw, match_ratio
from CNN import CNN



def train():
    train_data, train_label, test_data, test_label = load_MNIST()
    lr = 1e-4
    Epoch = 1
    Batchsize_test = 20
    Batchsize_train = 600
    Iteration = len(train_data) // Batchsize_train

    train_data, train_label, test_data, test_label = load_MNIST()

    cnn = CNN()

    optimizer = optim.Adam(cnn.parameters(), lr=lr)   # optimize all cnn parameters
    loss_func = nn.CrossEntropyLoss()   # the target label is not one-hotted


    for epoch in range(Epoch):
         for j in range(Iteration):   # 分配 batch data, normalize x when iterate train_loader
             x_train, x_label= random_draw(train_data,train_label,Batchsize_train)
             x_train1 = torch.from_numpy(x_train).to(torch.float32)
             x_train11 = x_train1.reshape(-1,28*28)

             output = cnn(x_train11)               # cnn output
             loss = loss_func(output, x_label)   # cross entropy loss
             optimizer.zero_grad()           # clear gradients for this training step
             loss.backward()                 # backpropagation, compute gradients
             optimizer.step()                # apply gradients

         print("epoch = %d, loss = %.4f, corret rate = %.2f" % (Epoch, loss, match_ratio(output, x_label)))

    x_test, t_label = random_draw(test_data, test_label, Batchsize_test)
    result = cnn(x_test)
    loss = loss_func(result, t_label)
    print('After Training.\nTest loss = %.4f, correct rate = %.3f' % (loss, match_ratio(result, t_label)))




if __name__ == "__main__":
    train()
