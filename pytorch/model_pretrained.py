"""
现有网络模型的使用及修改
"""

import torchvision
from torch import nn

#train_data = torchvision.datasets.ImageNet("data_image_net",split='train',download=True,transform=torchvision.transforms.ToTensor())   #无法公开访问，数据集过大

#网络模型，vgg模型最后一层是1000个分类
vgg16_false = torchvision.models.vgg16(pretrained=False)      #没有进行训练的参数
vgg16_true = torchvision.models.vgg16(pretrained=True)      #进行训练过的参数

#数据集
train_data = torchvision.datasets.CIFAR10('dataset',train=True,transform=torchvision.transforms.ToTensor(),download=True)   #分类类型为10

#添加网络模型层次
vgg16_true.add_module('add_linear',nn.Linear(1000,10))   #加在了vgg16的下一个层级
vgg16_true.classifier.add_module('add_linear',nn.Linear(1000,10))   #加在vgg16的内部classifier层级

#修改最后一层线性层
vgg16_false.classifier[6] = nn.Linear(4096,10)

print(vgg16_true)