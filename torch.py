'''
Author: Travis Chen cxqlove1999@outlook.com
Date: 2022-08-02 18:55:49
LastEditors: Travis Chen cxqlove1999@outlook.com
LastEditTime: 2023-02-15 08:58:24
FilePath: /t2l/torch.py
Description: 

Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
'''

#######################################
##    本包用于存储学习DL中的各种所需函数  ##
##      by   chenxuqiang             ##
#######################################

import inspect
import torch
import torchvision
from torch.utils import data
from torchvision import transforms

from torch import nn
import numpy as np
import time
import collections
import math
import shutil
import random
import sys
import os


from IPython import display
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline

t2l = sys.modules[__name__]


def test():
    print("Hello t2l")


def synthetic_data(w, b, num_examples):
    """

    """
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


def use_svg_display():
    """Use the svg format to display a plot in Jupyter.

    Defined in :numref:`sec_calculus`"""
    backend_inline.set_matplotlib_formats('svg')


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib.

    Defined in :numref:`sec_calculus`"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


def set_figsize(figsize=(3.5, 2.5)):
    """Set the figure size for matplotlib.

    Defined in :numref:`sec_calculus`"""
    use_svg_display()
    t2l.plt.rcParams['figure.figsize'] = figsize


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """
    输入：
      imgs： 图片的列表，可以是图片也可以是像素值
      num_rows and num_cols: 列数以及行数
      titles： 标题，可以是一个列表
      scale： 放大倍数
    绘制图像列表
    """
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = t2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


######### 常用组件 ########

def get_fashion_mnist_labels(labels):  # @save
    """
    输入：数字列表
    返回：Fashion-MNIST数据集的文本标签的列表
    """
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def data_iter(batch_size, features, labels):
    """
    mini-batch
    """
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


def get_dataloader_workers():
    """
    使用4个进程来读取数据
    输出：需要的进程数
    """
    return 4


def load_array(data_arrays, batch_size, is_train=True):  # @save
    """
      构造一个PyTorch数据迭代器
    """
    # 将数据转换为tensor
    dataset = torch.utils.data.TensorDataset(*data_arrays)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=is_train)


def load_data_fashion_mnist(batch_size, resize=None):
    """
    下载Fashion-MNIST数据集，然后将其加载到内存中
    用于获取和读取Fashion-MNIST数据集。这个函数返回 训练集和验证集的数据迭代器。此外，这个函数还接受一个可选参数resize，用来将图像大小调整为另一 种形状。
    返回两个iter分别是训练集和测试集
    iter是循环器，每次循环给X，y，其形状为【minibatch，channel，】
    """
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers()), data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=get_dataloader_workers()))


def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])


def accuracy(y_hat, y):  # @save
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        # y_hat第二维是每一类的分数，argmax输出最大的是哪一类，
        y_hat = y_hat.argmax(axis=1)
    # 然后通过比较y和y_hat, 之后返回预测正确的数量
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
        metric = Accumulator(2)  # 正确预测数、预测总数
        with torch.no_grad():
            for X, y in data_iter:
                metric.add(accuracy(net(X), y), y.numel())
        return metric[0] / metric[1]


### 相关类###

class Timer:
    """Record multiple running times."""

    def __init__(self):
        """Defined in :numref:`subsec_linear_model`"""
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()


class Accumulator:
    """在n个变量上累加"""

    def __init__(self, n):
        # 定义一个包含n个数据的列表
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def avg(self):
        return [sum(a)/len(a) for a in self.data]

    def __getitem__(self, idx):
        return self.data[idx]


class Animator:
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None, xscale='linear', yscale='linear', fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1, figsize=(3.5, 2.5)):
        # Incrementally plot multiple lines
        if legend is None:
            legend = []
        t2l.use_svg_display()
        self.fig, self.axes = t2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # Use a lambda function to capture arguments
        self.config_axes = lambda: t2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)


def train_epoch_ch3(net, train_iter, loss, updater):
    """训练模型一个迭代周期(定义⻅第3章)"""
# 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """Train a model (defined in Chapter 3).

    Defined in :numref:`sec_softmax_scratch`"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc


### 常见的损失函数 ##
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)


### 常见的模型 ###

def linreg(X, w, b):
    """The linear regression model.

    Defined in :numref:`sec_linear_scratch`"""
    return t2l.matmul(X, w) + b


def squared_loss(y_hat, y):
    """Squared loss.
    Defined in :numref:`sec_linear_scratch`"""
    return (y_hat - t2l.reshape(y, y_hat.shape)) ** 2 / 2


def l2_penalty(w):
    """
    定义L2范数，返回L2范数的w
    """
    return torch.sum(w.pow(2)) / 2


def init_params(num_inputs):
    """
    返回一个w和b的列表
    """
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]


def train(lambd, train_iter, test_iter, batch_size):
    w, b = init_params()
    net, loss = lambda X: t2l.linreg(X, w, b), t2l.squared_loss
    num_epochs, lr = 100, 0.003
    animator = t2l.Animator(xlabel='epochs', ylabel='loss', yscale='log', xlim=[
                            5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            # 增加了L2范数惩罚项，
            # 广播机制使l2_penalty(w)成为一个⻓度为batch_size的向量
            l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            t2l.sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (t2l.evaluate_loss(net, train_iter,
                         loss), t2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2范数是:', torch.norm(w).item())


class MySequential(torch.nn.Module):
    def __init__(self, *args):
        super.__init__()
        for idx, module in enumerate(args):
            # 这里，module是Module子类的一个实例。我们把它保存在'Module'类的成员 # 变量_modules中。_module的类型是OrderedDict
            self._modules[str(idx)] = module

    def forward(self, x):
        for block in self._modules.values():
            x = block(x)
        return x


def try_all_gpus():
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]"""
    devices = [torch.device(f'cuda:{i}')
               for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def read_csv_labels(fname):
    """读取fname来给标签字典返回一个文件名"""
    with open(fname, 'r') as f:
        # 跳过文件头行(列名)
        lines = f.readlines()[1:]
        tokens = [l.rstrip().split(',') for l in lines]
        return dict(((name, label) for name, label in tokens))


def copyfile(filename, target_dir):
    """将文件复制到目标目录"""
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(filename, target_dir)


def reorg_train_valid(data_dir, labels, valid_ratio):
    """将验证集从原始的训练集中拆分出来"""
    # 训练数据集中样本最少的类别中的样本数
    n = collections.Counter(labels.values()).most_common()[-1][1]
    # 验证集中每个类别的样本数
    n_valid_per_label = max(1, math.floor(n * valid_ratio))
    label_count = {}
    for train_file in os.listdir(os.path.join(data_dir, 'train')):
        label = labels[train_file.split('.')[0]]
        fname = os.path.join(data_dir, 'train', train_file)
        copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                     'train_valid', label))
    if label not in label_count or label_count[label] < n_valid_per_label:
        copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                     'valid', label))
        label_count[label] = label_count.get(label, 0) + 1
    else:
        copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                     'train', label))
    return n_valid_per_label


def reorg_test(data_dir):
    """在预测期间整理测试集，以方便读取"""
    for test_file in os.listdir(os.path.join(data_dir, 'test')):
        copyfile(os.path.join(data_dir, 'test', test_file),
                 os.path.join(data_dir, 'train_valid_test', 'test',
                              'unknown'))


def reorg_cifar10_data(data_dir, valid_ratio):
    """
    调用前面定义的函数整理数据
    """
    labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
    reorg_train_valid(data_dir, labels, valid_ratio)
    reorg_test(data_dir)

### #### ### ### ### ### #### ### ### ### ### #### ### ### ### ### #### ### ### ###


def resnet50(num_classes=10):
    """A slightly modified ResNet-18 model.

    Defined in :numref:`sec_multi_gpu_concise`"""
    # todo Bottleneck
    class Bottleneck(nn.Module):
        """
        __init__
            in_channel：残差块输入通道数
            out_channel：残差块输出通道数
            stride：卷积步长
            downsample：在_make_layer函数中赋值，用于控制shortcut图片下采样 H/2 W/2
        """
        expansion = 4   # 残差块第3个卷积层的通道膨胀倍率

        def __init__(self, in_channel, out_channel, stride=1, downsample=None):
            super(Bottleneck, self).__init__()
            self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                                   kernel_size=1, stride=1, bias=False)   # H,W不变。C: in_channel -> out_channel
            self.bn1 = nn.BatchNorm2d(num_features=out_channel)
            self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                                   kernel_size=3, stride=stride, bias=False, padding=1)  # H/2，W/2。C不变
            self.bn2 = nn.BatchNorm2d(num_features=out_channel)
            self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel*self.expansion,
                                   kernel_size=1, stride=1, bias=False)   # H,W不变。C: out_channel -> 4*out_channel
            self.bn3 = nn.BatchNorm2d(num_features=out_channel*self.expansion)
            self.relu = nn.ReLU(inplace=True)
            self.downsample = downsample

        def forward(self, x):
            identity = x    # 将原始输入暂存为shortcut的输出
            if self.downsample is not None:
                # 如果需要下采样，那么shortcut后:H/2，W/2。C: out_channel -> 4*out_channel(见ResNet中的downsample实现)
                identity = self.downsample(x)

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)

            out += identity     # 残差连接
            out = self.relu(out)

            return out

    # todo ResNet
    class ResNet(nn.Module):
        """
        __init__
            block: 堆叠的基本模块
            block_num: 基本模块堆叠个数,是一个list,对于resnet50=[3,4,6,3]
            num_classes: 全连接之后的分类特征维度

        _make_layer
            block: 堆叠的基本模块
            channel: 每个stage中堆叠模块的第一个卷积的卷积核个数，对resnet50分别是:64,128,256,512
            block_num: 当期stage堆叠block个数
            stride: 默认卷积步长
        """

        def __init__(self, block, block_num, num_classes=1000):
            super(ResNet, self).__init__()
            self.in_channel = 64    # conv1的输出维度

            self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.in_channel,
                                   kernel_size=7, stride=2, padding=3, bias=False)     # H/2,W/2。C:3->64
            self.bn1 = nn.BatchNorm2d(self.in_channel)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(
                kernel_size=3, stride=2, padding=1)     # H/2,W/2。C不变
            # H,W不变。downsample控制的shortcut，out_channel=64x4=256
            self.layer1 = self._make_layer(
                block=block, channel=64, block_num=block_num[0], stride=1)
            # H/2, W/2。downsample控制的shortcut，out_channel=128x4=512
            self.layer2 = self._make_layer(
                block=block, channel=128, block_num=block_num[1], stride=2)
            # H/2, W/2。downsample控制的shortcut，out_channel=256x4=1024
            self.layer3 = self._make_layer(
                block=block, channel=256, block_num=block_num[2], stride=2)
            # H/2, W/2。downsample控制的shortcut，out_channel=512x4=2048
            self.layer4 = self._make_layer(
                block=block, channel=512, block_num=block_num[3], stride=2)

            # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(
                in_features=512*block.expansion, out_features=num_classes)

            for m in self.modules():    # 权重初始化
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode='fan_out', nonlinearity='relu')

        def _make_layer(self, block, channel, block_num, stride=1):
            downsample = None   # 用于控制shorcut路的
            # 对resnet50：conv2中特征图尺寸H,W不需要下采样/2，但是通道数x4，因此shortcut通道数也需要x4。对其余conv3,4,5，既要特征图尺寸H,W/2，又要shortcut维度x4
            if stride != 1 or self.in_channel != channel*block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(in_channels=self.in_channel, out_channels=channel*block.expansion,
                              kernel_size=1, stride=stride, bias=False),  # out_channels决定输出通道数x4，stride决定特征图尺寸H,W/2
                    nn.BatchNorm2d(num_features=channel*block.expansion))

            layers = []  # 每一个convi_x的结构保存在一个layers列表中，i={2,3,4,5}
            # 定义convi_x中的第一个残差块，只有第一个需要设置downsample和stride
            layers.append(block(in_channel=self.in_channel,
                          out_channel=channel, downsample=downsample, stride=stride))
            # 在下一次调用_make_layer函数的时候，self.in_channel已经x4
            self.in_channel = channel*block.expansion

            for _ in range(1, block_num):  # 通过循环堆叠其余残差块(堆叠了剩余的block_num-1个)
                layers.append(
                    block(in_channel=self.in_channel, out_channel=channel))

            return nn.Sequential(*layers)   # '*'的作用是将list转换为非关键字参数传入

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

            return x

    return ResNet(block=Bottleneck, block_num=[3, 4, 6, 3], num_classes=num_classes)


Alexnet = nn. Sequential(
    # 使用一个11*11 的更大的窗口来捕捉对象
    # 同时步幅为4，以减少输出的宽度和高度
    # 另外，输出通道远大于LeNet

    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),

    # 减小卷积窗口(原来为11)，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 使用三个连续的卷积层和较小的卷积窗口
    # 除了最后的卷积层， 输出通道的数量进一步增加
    # 在前两个卷积层之后， 汇聚层不用雨减少输入的高度和宽度
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    # 这里，全连接层的输出数量是LeNet中的好几倍。使用dropout层来减轻过拟合

    nn.Linear(6400, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),

    nn.Linear(4096, 10)
)


### #### ### ### ### ### #### ### ### ### ### #### ### ### ### ### #### ### ### ###


################################# 训练函数#################################
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """Train a model with a GPU (defined in Chapter 6).

    Defined in :numref:`sec_lenet`"""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = t2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = t2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples
        metric = t2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], t2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')


def evaluate_accuracy_gpu(net, data_iter, device=None):  # @save """使用GPU计算模型在数据集上的精度"""

    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device  # 正确预测的数量，总预测的数量
    metric = t2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
             # BERT微调所需的(之后将介绍)
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(t2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def vgg_block(num_conv, in_channels, out_channels):
    '''
    input:
    - num_conv: 卷积层数量
    - in_channel: 输入通道数
    - out_channel: 输出通道数
    output:
    - 对应的网络
    '''
    layers = []
    for _ in range(num_conv):
        layers.append(nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


def vgg11(conv_arch):
    conv_blks = []
    in_channels = 1
    # 卷积层部分
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        # 全联接层
        nn.Linear(out_channels*7*7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(),
        nn.Linear(4096, 10)
    )


def used_wandb(lass):
    import wandb
    from tqdm import tqdm
    def train(net, train_features, train_labels, test_features, test_labels,
              num_epochs, learning_rate, weight_decay, batch_size, device):
        wandb.watch(net)
        train_ls, test_ls = [], []
        train_iter = load_array((train_features, train_labels), batch_size)
        # 这里使用的是Adam优化算法
        optimizer = torch.optim.Adam(
            net.parameters(), lr=learning_rate, weight_decay=weight_decay)
        for epoch in tqdm(range(num_epochs)):
            for X, y in train_iter:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                outputs = net(X)
                loss = loss(outputs, y)
                loss.backward()
                optimizer.step()
            record_loss = loss(net.to('cpu'), train_features, train_labels)
            wandb.log({'loss': record_loss, 'epoch': epoch})
            train_ls.append(record_loss)
            net.to(device)
        wandb.finish()
        return train_ls, test_ls

    k, num_epochs, lr, weight_decay, batch_size = 5, 2000, 0.005, 0.05, 256
    wandb.init(project="kaggle_predict",
               config={"learning_rate": lr,
                       "weight_decay": weight_decay,
                       "batch_size": batch_size,
                       "total_run": num_epochs
                       }
               )
    return lass


def how_to_plot():
    import matplotlib.pyplot as plt

    plt.rcParams['font.family'] = 'sans-serif'  # 用来正常显示中文
    plt.rcParams['font.sans-serif'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False   # 设置正常显示符号

    # 绘制直方图
    plt.hist(np.random.rand(100000), density=True, bins=100,
             histtype="step", color="blue", label="rand")
    plt.hist(np.random.randn(100000), density=True, bins=100,
             histtype="step", color="red", label="randn")
    # 设置坐标轴的范围
    plt.axis([-2.5, 2.5, 0, 1.1])
    # 设置图例在左上角
    plt.legend(loc="upper left")
    # 设置标题
    plt.title("随机值分布")
    # 设置x轴
    plt.xlabel("值")
    # 设置y轴
    plt.ylabel("密度")
    # 显示
    plt.show()


############ 自用函数#########

############### Out #########################
# def get_variable_name(loc, variable):
#     """
#     Get the name of variable
#     loc: locals()
#     variable: the variable that you want to get the name
#     """
#     for k, v in loc.items():
#         if loc[k] is variable:
#             return k
#     raise NotImplementedError


def retrieve_name(var):
    """
    获取名称, 如果嵌套需要改变.f_back的数量
    """
    callers_local_vars = inspect.currentframe().f_back.f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]


def printvariable(var, n=None):
    """
    name: optional name
    easy to print the variable with its name
    Not support method use
    """
    if n:
        name = n
    else:
        name = retrieve_name(var)
    # name = get_variable_name(loc, var)
    print(f'name: {name} | value: {var}\n')


def printc(n=50):
    print(n*'-')
