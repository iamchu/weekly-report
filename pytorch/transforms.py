from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs")
img_path = "hymenoptera_data/train/ants/45472593_bfd624f8dc.jpg"
img = Image.open(img_path)

#ToTensor
#将工具具体化,将图片转换为张量
tensor_trans = transforms.ToTensor()
#使用工具
tensor_img = tensor_trans(img)  #自动归一化到（0-1）
writer.add_image("ToTensor",tensor_img)

#Normalize
print(tensor_img[0][0][0])
trans_norm = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]) #括号里的参数是均值和标准差（三个通道）
norm_img = trans_norm(tensor_img)
print(norm_img[0][0][0])
writer.add_image("Normalize",norm_img,1)

#Resize
trans_resize = transforms.Resize((512,512))
img_resize = trans_resize(img)
img_resize = tensor_trans(img_resize)
writer.add_image("Resize",img_resize,0)

#Compose
trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize_2,tensor_trans])
img_resize_2 = trans_compose(img)
writer.add_image("Resize",img_resize_2,1)

#RandomCrop
trans_random = transforms.RandomCrop(333)
trans_compose_2 = transforms.Compose([trans_random,tensor_trans])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop",img_crop,i)

writer.close()