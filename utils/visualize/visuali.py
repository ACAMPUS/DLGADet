import os.path

import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import cv2, imageio


# 将输入图片转化为卷积运算格式tensor数据
def transfer_image(image_dir):
    # 以RGB格式打开图像
    # Pytorch DataLoader就是使用PIL所读取的图像格式
    # 建议就用这种方法读取图像，当读入灰度图像时convert('')
    image_info = Image.open(image_dir).convert('RGB')
    # 数据预处理方法
    image_transform = transforms.Compose([
        transforms.Resize(1024),  # 放缩图片大小至1024*1024
        transforms.CenterCrop(672),  # 从图象中间剪切672*672大小
        transforms.ToTensor(),  # 将图像数据转化为tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化处理
    ])
    image_info = image_transform(image_info)  # 更新载入图像的数据格式
    # 下面这部分是tensor张量转化为图片的过程，opencv画图和PIL、torch画图方式
    # array1 = image_info.numpy()
    # maxvalue = array1.max()
    # array1 = array1*255 / maxvalue
    # mat = np.uint8(array1)
    # print('mat_shape',mat.shape)
    # mat = mat.transpose(1,2,0)
    # cv2.imwrite('convert_cocomi1.jpg',mat)
    # mat=cv2.cvtColor(mat,cv2.COLOR_BGR2RGB)
    # cv2.imwrite('convert_cocomi2.jpg',mat)

    image_info = image_info.unsqueeze(0)  # 增加tensor维度，便于卷积层进行运算
    return image_info


# 获取第k层的特征图
def get_k_layer_feature_map(feature_extractor, k, x):
    with torch.no_grad():
        # feature_extractor是特征提取层，后面可以具体看一下vgg16网络
        for index, layer in enumerate(feature_extractor):
            x = layer(x)  # x是输入图像的张量数据，layer是该位置进行运算的卷积层，就是进行特征提取
            # print('k values:',k)
            # print('feature_extractor layer:',index)
            if k == index:  # k代表想看第几层的特征图

                return x


#  可视化特征图
def show_feature_map(feature_map, k):
    if not os.path.exists(f'../images/{k}/'):
        os.mkdir(f'images/{k}')
    feature_map = feature_map.squeeze(0)  # squeeze(0)实现tensor降维，开始将数据转化为图像格式显示
    feature_map = feature_map.cpu().numpy()  # 进行卷积运算后转化为numpy格式
    feature_map_num = feature_map.shape[0]  # 特征图数量等于该层卷积运算后的特征图维度
    # row_num = np.ceil(np.sqrt(feature_map_num))
    # plt.figure()
    for index in range(1, feature_map_num + 1):
        fw = np.int64(feature_map[index - 1] > 0)
        imageio.imwrite("images/" + str(k) + '/' + str(index) + ".png", fw)
    # plt.show()


if __name__ == '__main__':
    # 初始化图像的路径 path=r'D:\dataset\data\my_tt100k\train\images\23.jpg'
    image_dir = r'E:\paper\code\yolov5-6.1\utils\visualize\images\10009.png'
    # 定义提取第几层的feature map
    k = 1
    # 导入Pytorch封装的vgg16网络模型
    model = models.vgg16(pretrained=False)
    # 是否使用gpu运算
    # use_gpu = torch.cuda.is_available()
    use_gpu = False
    # 读取图像信息
    image_info = transfer_image(image_dir)
    # 判断是否使用gpu
    if use_gpu:
        model = model.cuda()
        image_info = image_info.cuda()
    # vgg16包含features和classifier,但只有features部分有特征图
    # classifier部分的feature map是向量
    feature_extractor = model.features
    feature_map = get_k_layer_feature_map(feature_extractor, k, image_info)
    show_feature_map(feature_map, k)
