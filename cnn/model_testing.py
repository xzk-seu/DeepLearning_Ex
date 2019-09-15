import torch
from cnn.visualization import last_layer_display
from cnn.data_preprocess import BATCH_SIZE


torch.manual_seed(1)    # reproducible


def model_testing(cnn, test_loader):
    total_correct = 0
    test_round = 0
    last_layer = None
    test_y = None
    for data in test_loader:
        test_x, test_y = data
        test_x = test_x.cuda()
        test_y = test_y.cuda()
        test_output, last_layer = cnn(test_x)
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





