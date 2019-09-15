import os
import torch
import torch.nn as nn
from datetime import datetime
from cnn.model import CNN
from cnn.data_preprocess import get_train_loader, get_test_loader, MODEL_PATH
from cnn.model_testing import model_testing


torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 1           # 训练整批数据多少次, 为了节约时间, 我们只训练一次
LR = 0.001          # 学习率


def train(cnn, train_loader):
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # optimize all cnn parameters
    loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted
    # training
    for epoch in range(EPOCH):
        for step, (b_x, b_y) in enumerate(train_loader):  # 分配 batch data, normalize x when iterate train_loader
            b_x = b_x.cuda()
            b_y = b_y.cuda()
            output, _ = cnn(b_x)  # cnn output
            loss = loss_func(output, b_y)  # cross entropy loss
            """
            # output size [100, 10]=[batch_size, num_labels]
            # b_y size = [100] = batch_size
            """
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # back-propagation, compute gradients
            optimizer.step()  # apply gradients

            if step % 50 == 0:
                loss = loss.cpu()
                print('Epoch: ', epoch, '| Step: ', step, '| train loss: %.4f ' % loss.data.numpy(),
                      '|  time: %s' % datetime.now())


if __name__ == '__main__':
    trainloader = get_train_loader()
    testloader = get_test_loader()
    cnn1 = CNN()
    cnn1.cuda()
    train(cnn1, trainloader)
    model_testing(cnn1, testloader)
    model_path = os.path.join(MODEL_PATH, 'cnn1.pkl')
    torch.save(cnn1, model_path)  # 保存整个网络
