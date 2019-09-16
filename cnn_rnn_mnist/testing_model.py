import torch
from cnn_rnn_mnist.visualization import last_layer_display
from cnn_rnn_mnist.data_load import BATCH_SIZE


torch.manual_seed(1)    # reproducible


def model_testing(net, test_loader, is_rnn):
    total_correct = 0
    test_round = 0
    last_layer = None
    test_y = None
    for data in test_loader:
        test_x, test_y = data
        test_x = test_x.cuda()
        if is_rnn:
            test_x = test_x.view(-1, 28, 28)  # reshape x to (batch, time_step, input_size)
            # cnn需要chanel，输入为4阶张量，rnn不需要，所以输入为三阶张量
        test_y = test_y.cuda()
        test_output, last_layer = net(test_x)
        _, pred_y = torch.max(test_output, dim=1)  # 在第一个维度上的最大值以及索引
        pred_y = pred_y.cuda()
        count = (pred_y == test_y)
        correct = count.sum()
        total_correct += int(correct)
        test_round += 1
    acc = float(total_correct) / float(test_round * BATCH_SIZE)
    print('test accuracy: %.2f' % acc)
    last_layer_display(last_layer, test_y)
    return acc





