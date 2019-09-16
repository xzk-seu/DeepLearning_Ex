import torch
import torch.nn as nn
from datetime import datetime


torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 1           # 训练整批数据多少次, 为了节约时间, 我们只训练一次
LR = 0.001          # 学习率


def train(net, train_loader, is_rnn):
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)  # optimize all cnn_rnn_mnist parameters
    loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted
    # training
    for epoch in range(EPOCH):
        for step, (b_x, b_y) in enumerate(train_loader):  # 分配 batch data, normalize x when iterate train_loader
            b_x = b_x.cuda()
            if is_rnn:
                b_x = b_x.view(-1, 28, 28)  # reshape x to (batch, time_step, input_size)
                # cnn需要chanel，输入为4阶张量，rnn不需要，所以输入为三阶张量
            b_y = b_y.cuda()
            output, _ = net(b_x)  # cnn_rnn_mnist output
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
