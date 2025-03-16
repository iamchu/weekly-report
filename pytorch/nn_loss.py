'''
loss function
为了计算实际输出和目标之间的差距
为我们反向传播更新输出提供一定依据
'''
#L1Loss
import torch
from torch.nn import L1Loss
from torch import nn

inputs = torch.tensor([1,2,3],dtype=torch.float32)    #要求用小数，实际上用的数据集一般都是小数
targets = torch.tensor([1,2,5],dtype=torch.float32)

inputs = torch.reshape(inputs,(1,1,1,3))
targets = torch.reshape(targets,(1,1,1,3))

loss = L1Loss(reduction='sum')
result = loss(inputs,targets)

#MSEloss
loss_mse = nn.MSELoss()
result_mse = loss_mse(inputs,targets)

#交叉熵损失函数
x = torch.tensor([0.1,0.2,0.3])
y = torch.tensor([1])
x = torch.reshape(x,(1,3))
loss_cross = nn.CrossEntropyLoss()
result_cross = loss_cross(x,y)

print(result)
print(result_mse)
print(result_cross)