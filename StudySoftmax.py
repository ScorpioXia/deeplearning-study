#% matplotlib inline
import torch
import torchvision
from IPython.core.pylabtools import figsize
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l
from torch import nn

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# PyTorch不会隐式地调整输入的形状。因此，
# 我们在线性层前定义了展平层（flatten），来调整网络输入的形状
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10)) # 定义网络

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);  # 重置权重

loss = nn.CrossEntropyLoss(reduction='none') # 设置loss

trainer = torch.optim.SGD(net.parameters(), lr=0.1) # 设置优化算法

num_epochs = 10

d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer) # 完成训练，这里是全部集成了的，以后可能要自己单独改。

"""
下面是数据初始化的练习
"""
# from testone.StudyLinearRegression import batch_size

# d2l.use_svg_display()
#
# trans = transforms.ToTensor()
# mnist_train = torchvision.datasets.FashionMNIST(
#     root="../data", train=True, transform=trans
# )
# mnist_test = torchvision.datasets.FashionMNIST(
#     root="../data", train=False, transform=trans
# )

#len(mnist_train), len(mnist_test) #直接写，就类似print，但是在交互式变成里面才有用

#mnist_train[0][0].shape
#
# def get_fashion_mnist_labels(labels): #@save
#     """返回Fashion-MNIST数据集的文本标签"""
#     text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
#     return [text_labels[int(i)] for i in labels]
#
# def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
#     """绘制图像列表"""
#     figsize = (num_cols * scale, num_rows * scale)
#     _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
#     axes = axes.flatten()
#     for i, (ax, img) in enumerate(zip(axes, imgs)):
#         if torch.is_tensor(img):
#             # 图片张量
#             ax.imshow(img.numpy())
#         else:
#             # PIL图片
#             ax.imshow(img)
#         ax.axes.get_xaxis().set_visible(False)
#         ax.axes.get_yaxis().set_visible(False)
#         if titles:
#             ax.set_title(titles[i])
#     return axes
#
# # X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
# # show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y));
#
# def get_dataloader_workers(): #@save
#     """使用4个进程来读取数据"""
#     return 0

# train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers())
#
# timer = d2l.Timer()
# for X, y in train_iter:
#     continue

# f'{timer.stop():.2f} sec' #交互式编程时可以起到类似print的作用

# def load_data_fashion_mnist(batch_size, resize=None):  #@save
#     """下载Fashion-MNIST数据集，然后将其加载到内存中"""
#     trans = [transforms.ToTensor()]
#     if resize:
#         trans.insert(0, transforms.Resize(resize))
#     trans = transforms.Compose(trans)
#     mnist_train = torchvision.datasets.FashionMNIST(
#         root="../data", train=True, transform=trans)
#     mnist_test = torchvision.datasets.FashionMNIST(
#         root="../data", train=False, transform=trans)
#     return (data.DataLoader(mnist_train, batch_size, shuffle=True,
#                             num_workers=get_dataloader_workers()),
#             data.DataLoader(mnist_test, batch_size, shuffle=False,
#                             num_workers=get_dataloader_workers()))

# train_iter, test_iter = load_data_fashion_mnist(32, resize=64)

# for X, y in train_iter:
#     print(X.shape, X.dtype, y.shape, y.dtype)
#     break