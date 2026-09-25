#%matplotlib inline
import random
import torch
import matplotlib.pyplot as plt

def synthetic_data(w, b, num_examples):  #@save
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

def linreg(X, w, b):  #@save
    """线性回归模型"""
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y):  #@save
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(params, lr, batch_size):  #@save
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

def synthetic_data(w, b, num_examples):
    """生成 y = Xw + b + 噪声 的数据集。"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0, y.shape) # 添加一些标准差为0.01的噪声
    return X, y.reshape((-1, 1))


if __name__=='__main__':
    print("Hello, world!")
    #我们通过从均值为0、标准差为0.01的正态分布中采样随机数来初始化权重，并将偏置初始化为0。
    w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)  
    print("w:", w)
    print("b:", b)

    lr = 0.03
    num_epochs = 3
    net = linreg
    loss = squared_loss

    batch_size = 10

    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)
    print('features:', features[0],'\nlabel:', labels[0])

    # 3. 创建画布（1行2列的子图）
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 绘制特征 1 与标签的关系
    axes[0].scatter(features[:, 0].numpy(), labels.numpy(), s=1)
    axes[0].set_title("Feature 1 vs Labels")
    axes[0].set_xlabel("features[:, 0]")
    axes[0].set_ylabel("labels")
    axes[0].grid(True)

    # 绘制特征 2 与标签的关系
    axes[1].scatter(features[:, 1].numpy(), labels.numpy(), s=1)
    axes[1].set_title("Feature 2 vs Labels")
    axes[1].set_xlabel("features[:, 1]")
    axes[1].set_ylabel("labels")
    axes[1].grid(True)

    # 4. 调整布局并显示图表
    plt.tight_layout()
    plt.show()
    for X, y in data_iter(batch_size, features, labels):
        print(X, '\n', y)
        break

    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            l = loss(net(X, w, b), y)  # X和y的小批量损失
            # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
            # 并以此计算关于[w,b]的梯度
            l.sum().backward()
            sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
        with torch.no_grad():
            train_l = loss(net(features, w, b), labels)
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
            print(f'epoch {epoch + 1}, w: {w.reshape(true_w.shape)}, b: {b}')

