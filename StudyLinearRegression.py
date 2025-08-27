import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

# print('features:', features[0],'\nlabel:', labels[0])

def load_array(data_arrays, batch_size, is_train=True): #@save
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

# next(iter(data_iter))
# iter() 将 data_iter 转换为迭代器，然后通过 next() 获取迭代器的下一个元素，即一个批次的特征和标签数据。通俗来讲，next就是抽一批数据。

net = nn.Sequential(nn.Linear(2, 1))
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
loss = nn.MSELoss()

trainer = torch.optim.SGD(net.parameters(), lr=0.03)
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y) #计算loss，这个loss没有观测价值，只是一个epoch里面一小部分数据的loss，对整个模型参数的学习没有指导意义
        trainer.zero_grad() #梯度清零
        l.backward()    #梯度反传
        trainer.step()  #更新参数，训练模型的关键步骤
    l = loss(net(features), labels) #计算一个epoch跑完所有数据后的loss
    print(f'epoch {epoch + 1}, loss {l:f}')

