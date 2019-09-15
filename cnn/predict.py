import os
import torch
from cnn.data_preprocess import get_predict_data, MODEL_PATH
from cnn.visualization import get_digit_picture


torch.manual_seed(1)    # reproducible


def predict(cnn, data_num):
    data = get_predict_data(data_num)
    for n, datum in enumerate(data):
        x, y = datum  # x为1 * 28 * 28的一张图片， y为一个整数
        input_x = x.unsqueeze(dim=0)  # 由于只能按批次输出，因此需要把一张图片作为一批，tensor shape为[1, 1, 28, 28]
        test_output, last_layer = cnn(input_x.cuda())
        _, pred_y = torch.max(test_output, dim=1)  # 在第一个维度上的最大值以及索引
        pred_y = int(pred_y)

        get_digit_picture(x.squeeze(), y)
        print('picture: #%d | this number is %d | prediction is %d\n' % (n, y, pred_y))


if __name__ == '__main__':
    model_path = os.path.join(MODEL_PATH, 'cnn1.pkl')
    cnn1 = torch.load(model_path)
    predict(cnn1, 10)
