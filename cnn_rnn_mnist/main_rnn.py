import os
import torch
from datetime import datetime
from cnn_rnn_mnist.model_rnn import RNN
from cnn_rnn_mnist.data_load import get_train_loader, get_test_loader, MODEL_PATH
from cnn_rnn_mnist.testing_model import model_testing
from cnn_rnn_mnist.train import train


torch.manual_seed(1)    # reproducible


if __name__ == '__main__':
    print(datetime.now(), 'START load data\n')
    trainloader = get_train_loader()
    testloader = get_test_loader()
    print(datetime.now(), 'START generate net\n')
    rnn1 = RNN()
    rnn1.cuda()
    print(datetime.now(), 'START train net\n')
    train(rnn1, trainloader, is_rnn=True)
    print(datetime.now(), 'START test net\n')
    model_testing(rnn1, testloader, is_rnn=True)
    print(datetime.now(), 'START save net\n')
    model_path = os.path.join(MODEL_PATH, 'rnn1.pkl')
    torch.save(rnn1, model_path)  # 保存整个网络
    print(datetime.now(), 'OVER\n')

