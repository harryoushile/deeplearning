
import numpy as np
import torch
from torch.utils import data
from torch import nn

def synthetic_data(w, b, num_examples):
    """生成 y = Xw + b + 噪声 的数据集。"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0, y.shape) # 添加一些标准差为0.01的噪声
    return X, y.reshape((-1, 1))


def load_array(data_arrays, batch_size, is_train=True):  #@save
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

if __name__=='__main__':
    print("Hello, world!")
    batch_size = 10
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)
    data_iter = load_array((features, labels), batch_size)

    next(iter(data_iter))
    # nn是神经网络的缩写
    net = nn.Sequential(nn.Linear(2, 1))
    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)
    loss = nn.MSELoss()
    trainer = torch.optim.SGD(net.parameters(), lr=0.03)
    num_epochs = 3
    for epoch in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X) ,y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        l = loss(net(features), labels)
        print(f'epoch {epoch + 1}, loss {l:f}')


    #比较生成数据集的真实参数和通过有限数据训练获得的模型参数
    w = net[0].weight.data
    print('w的估计误差：', true_w - w.reshape(true_w.shape))
    b = net[0].bias.data
    print('b的估计误差：', true_b - b)