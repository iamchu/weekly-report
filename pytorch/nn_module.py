from torch import nn
import torch

class Moxing(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,input):
        output = input + 1
        return output

Moxing_test = Moxing()
x = torch.tensor(1.0)
output = Moxing_test(x)
print(output)