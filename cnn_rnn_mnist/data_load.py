import os
import random
import torch
import torch.utils.data as data
import torchvision      # 数据库模块

torch.manual_seed(1)    # reproducible


MODEL_PATH = os.path.join(os.getcwd(), 'model_dump')
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)
BATCH_SIZE = 100
DOWNLOAD_MNIST = False
if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    DOWNLOAD_MNIST = True


def get_train_loader():
    # Mnist 手写数字
    train_data = torchvision.datasets.MNIST(
        root='./mnist/',    # 保存或者提取位置
        train=True,  # this is training data
        transform=torchvision.transforms.ToTensor(),    # 转换 PIL.Image or numpy.ndarray 成
                                                        # torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
        download=DOWNLOAD_MNIST,          # 没下载就下载, 下载了就不用再下了
    )
    # 批训练 50samples, 1 channel, 28x28 (50, 1, 28, 28)
    train_loader = data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    print('train_data size is: ', train_data.data.size())  # (60000, 28, 28)
    print('train_labels size is: ', train_data.targets.size())  # (60000)
    return train_loader


def get_test_loader():
    test_data = torchvision.datasets.MNIST(root='./mnist/', train=False,
                                           transform=torchvision.transforms.ToTensor())
    test_loader = data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    print('test_data size is: ', test_data.data.size())
    print('test_labels size is: ', test_data.targets.size())
    return test_loader


def get_predict_data(num_of_data):
    predict_data = torchvision.datasets.MNIST(root='./mnist/', transform=torchvision.transforms.ToTensor())
    predict_data = random.choices(predict_data, k=num_of_data)
    return predict_data
