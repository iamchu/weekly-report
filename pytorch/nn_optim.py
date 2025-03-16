import torch.optim
import torchvision.datasets
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("dataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader = DataLoader(dataset,batch_size=64)

class train_module2(nn.Module):
    def __init__(self):
        super(train_module2,self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self,x):
        x = self.model1(x)
        return x

#交叉熵损失函数
loss = nn.CrossEntropyLoss()
#实例化
tudui = train_module2()
#优化器：随机梯度下降
optim = torch.optim.SGD(tudui.parameters(),lr=0.01)
for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs,targets = data
        outputs = tudui(imgs)
        result_loss = loss(outputs,targets)
        optim.zero_grad()   #参数设置为0
        result_loss.backward()   #反向传播，计算出每个参数的梯度
        optim.step()   #优化更新
        running_loss = running_loss + result_loss
    print(running_loss)