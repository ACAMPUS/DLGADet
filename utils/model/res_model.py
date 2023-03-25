import cv2
from torchvision.models import resnet34

model=resnet34()
print(model)
path=r'D:\dataset\data\my_tt100k\train\images\23.jpg'
input=cv2.imread(path)