import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

#引入数据集
dataset = torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)
#加载数据集
dataloader = DataLoader(dataset,batch_size=64,drop_last=True)

class linear(nn.Module):
    def __init__(self):
        super(linear,self).__init__()
        self.linear1 = nn.Linear(3072,10)

    def forward(self,input):
        output = self.linear1(input)
        return output

linear_test = linear()

for data in dataloader:
    imgs,targets = data
    print(imgs.shape)
    output = torch.flatten(imgs,start_dim=1) #start_dim=1保留 batch 维度
    print(output.shape)
    output = linear_test(output)
    print(output.shape)