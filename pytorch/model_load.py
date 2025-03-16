import torch
import torchvision

#保存方式1，加载模型
model = torch.load("vgg16_method1.pth")

#保存方式2，加载模型，恢复成网络模型
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
print(vgg16)

#陷阱，使用方式一保存模型，如果是自己定义的模型可能需要复制到下载模型的文件中，或者单独存到一个文件中，在当前文件用import导入