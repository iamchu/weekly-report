import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

#引入数据集
dataset = torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)
#加载数据集
dataloader = DataLoader(dataset,batch_size=64)

class Pool(nn.Module):
    def __init__(self):
        super(Pool,self).__init__()
        self.maxpool = MaxPool2d(kernel_size=3,ceil_mode=True)

    def forward(self,input):
        output = self.maxpool(input)
        return output

Pool_test = Pool()

writer = SummaryWriter("logs_maxpool")
step = 0
for data in dataloader:
    imgs,targets = data
    writer.add_images("input",imgs,step)
    output = Pool_test(imgs)
    writer.add_images("output",output,step)
    step = step + 1

writer.close()