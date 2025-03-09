from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

#SummaryWriter是一个类，对其实例化
writer = SummaryWriter("logs")
image_path = "hymenoptera_data/train/ants/0013035.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)

#添加图片
writer.add_image("test",img_array,1,dataformats='HWC')  #通道数在最后时需要设置参数dataformats='HWC'

#添加数
for i in range(100):
    writer.add_scalar("y=x",i,i)

writer.close()

#注：改tag改参数会新增一个新的图像，但是不改tag直接改参数，会在上一个事件当中生成图像，形成混乱
