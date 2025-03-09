import  torch
import torchvision
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

#引入数据集
dataset = torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)
#加载数据集
dataloader = DataLoader(dataset,batch_size=64)

class relu(nn.Module):
    def __init__(self):
        super(relu,self).__init__()
        self.relu1 = ReLU()
        self.sigmoid1 = Sigmoid()

    def forward(self,input):
        output = self.sigmoid1(input)
        return output

relu_test = relu()

writer = SummaryWriter("logs_relu")
step = 0
for data in dataloader:
    imgs,targets = data
    writer.add_images("input",imgs,step)
    output = relu_test(imgs)
    writer.add_images("output",output,step)
    step = step + 1

writer.close()
