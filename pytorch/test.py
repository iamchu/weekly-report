import torch
from PIL import Image
import torchvision
from torch import nn

image = Image.open("img/001.png")

"""
加上image = image.convert("RGB").
因为png格式是四个通道，处理RGB三个通道外，还有一个透明通道，调用image = image.convert("RGB")保留其颜色通道
加上这一步后，可以适应png,jpg各种格式的图片
"""
image = image.convert("RGB")

transfrom = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
                                            torchvision.transforms.ToTensor()])

image = transfrom(image)

class model(nn.Module):
    def __init__(self):
        super(model,self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,1,2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4,64),
            nn.Linear(64,10)
        )

    def forward(self,x):
        x = self.model(x)
        return x

Model = torch.load("Model_0.pth")
image = torch.reshape(image,(1,3,32,32))   #没有batchsize尺寸会不符合，要reshape
Model.eval()
with torch.no_grad():
    output = Model(image)

print(output.argmax(1))