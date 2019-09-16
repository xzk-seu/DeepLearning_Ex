import os
import torch
from datetime import datetime
from cnn_rnn_mnist.model_cnn import CNN
from cnn_rnn_mnist.data_load import get_train_loader, get_test_loader, MODEL_PATH
from cnn_rnn_mnist.testing_model import model_testing
from cnn_rnn_mnist.train import train


torch.manual_seed(1)    # reproducible


if __name__ == '__main__':
    print(datetime.now(), 'START load data\n')
    trainloader = get_train_loader()
    testloader = get_test_loader()
    print(datetime.now(), 'START generate net\n')
    cnn1 = CNN()
    cnn1.cuda()
    print(datetime.now(), 'START train net\n')
    train(cnn1, trainloader, is_rnn=False)
    print(datetime.now(), 'START test net\n')
    model_testing(cnn1, testloader, is_rnn=False)
    print(datetime.now(), 'START save net\n')
    model_path = os.path.join(MODEL_PATH, 'cnn1.pkl')
    torch.save(cnn1, model_path)  # 保存整个网络
    print(datetime.now(), 'OVER\n')
