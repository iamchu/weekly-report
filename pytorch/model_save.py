import torch
import torchvision

vgg16 = torchvision.models.vgg16(pretrained=False)
#保存方式1:保存了网络模型的结构和参数
torch.save(vgg16,"vgg16_method1.pth")

#保存方式2:只保存网络模型参数，保存成字典(官方推荐)
torch.save(vgg16.state_dict(),"vgg16_method2.pth")