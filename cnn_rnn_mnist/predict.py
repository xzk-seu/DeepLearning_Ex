import os
import torch
from datetime import datetime
from cnn_rnn_mnist.data_load import get_predict_data, MODEL_PATH
from cnn_rnn_mnist.visualization import get_digit_picture
import matplotlib.pyplot as plt


torch.manual_seed(1)    # reproducible


def net_predict(net, data_num, is_rnn):
    data = get_predict_data(data_num)
    plt.ion()
    for n, datum in enumerate(data):
        x, y = datum  # x为1 * 28 * 28的一张图片， y为一个整数
        input_x = x
        if not is_rnn:
            input_x = x.unsqueeze(dim=0)  # 由于只能按批次输出，因此需要把一张图片作为一批，
            # 对于CNN tensor shape为[1, 1, 28, 28]
        test_output, last_layer = net(input_x.cuda())
        _, pred_y = torch.max(test_output, dim=1)  # 在第一个维度上的最大值以及索引
        print('picture: #%d | this number is %d | prediction is %d\n' % (n, y, int(pred_y)))
        get_digit_picture(x.squeeze(), y)
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    t = input('cnn or rnn? default==cnn ')
    if t == 'rnn':
        print(datetime.now(), 'LOAD rnn1\n')
        model_path = os.path.join(MODEL_PATH, 'rnn1.pkl')
        rnn1 = torch.load(model_path)
        net_predict(rnn1, 10, is_rnn=True)
    else:
        print(datetime.now(), 'LOAD cnn1\n')
        model_path = os.path.join(MODEL_PATH, 'cnn1.pkl')
        cnn1 = torch.load(model_path)
        net_predict(cnn1, 10, is_rnn=False)
