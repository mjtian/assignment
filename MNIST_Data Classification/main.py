from __future__ import division
import torch
from torch import optim
import torch.nn as nn
import numpy as np
from utils import load_MNIST, random_draw
from CNN import CNN




def train():
    train_data, train_label, test_data, test_label = load_MNIST()
    lr = 1e-3
    Epoch =40
    Batchsize_test = 20
    Batchsize_train = 600
    Iteration = len(train_data) // Batchsize_train

    train_data, train_label, test_data, test_label = load_MNIST()
    cnn = CNN()
    optimizer = optim.Adam(cnn.parameters(), lr=lr)   # optimize all cnn parameters
    loss_func = nn.CrossEntropyLoss()   # the target label is not one-hotted
    TrainLOSS = []
    TestLOSS = []

    for epoch in range(Epoch):
         for j in range(Iteration):
             x_train, x_label= random_draw(train_data,train_label,Batchsize_train)
             x_train1 = torch.from_numpy(x_train).to(torch.float32)
             x_train11 = x_train1.reshape(-1,1,28,28)
             x_label1 = torch.from_numpy(x_label).to(torch.float32)
             x_label11 = torch.nonzero(x_label1).squeeze()
             x_label11 = torch.unbind(x_label11, 1)
             x_label111 = x_label11[1]


             output = cnn(x_train11)
             loss = loss_func(output, x_label111)
             # import pdb
             # pdb.set_trace()
             optimizer.zero_grad()
             loss.backward()
             optimizer.step()

             TrainLOSS.append(loss.item())
             x_test, x_testlabel= random_draw(test_data,test_label,Batchsize_test)
             x_test1 = torch.from_numpy(x_test).to(torch.float32)
             x_test11 = x_test1.reshape(-1,1,28,28)

             x_testlabel1 = torch.from_numpy(x_testlabel).to(torch.float32)
             x_testlabel11 = torch.nonzero(x_testlabel1).squeeze()
             x_testlabel11 = torch.unbind(x_testlabel11, 1)
             x_testlabel111 = x_testlabel11[1]
             output2 = cnn(x_test11)
             loss2 = loss_func(output2, x_testlabel111)
             TestLOSS.append(loss2.item())

         print("epoch = %d, loss = %.4f, test loss = %.4f" %(epoch, loss, loss2))
    trainLoss = np.array(TrainLOSS)
    testLoss = np.array(TestLOSS)
    from matplotlib import pyplot as plt
    plt.plot(trainLoss,label="Training")
    plt.plot(testLoss,label="Test")
    plt.legend()
    plt.show()


    x_test, x_testlabel= random_draw(test_data,test_label,Batchsize_test)
    x_test1 = torch.from_numpy(x_test).to(torch.float32)
    x_test11 = x_test1.reshape(-1,1,28,28)

    x_testlabel1 = torch.from_numpy(x_testlabel).to(torch.float32)
    x_testlabel11 = torch.nonzero(x_testlabel1).squeeze()
    x_testlabel11 = torch.unbind(x_testlabel11, 1)
    x_testlabel111 = x_testlabel11[1]
    output2 = cnn(x_test11)
    loss2 = loss_func(output2, x_testlabel111)
    print('After Training.\nTest loss = %.4f' % loss2)




if __name__ == "__main__":
    train()



