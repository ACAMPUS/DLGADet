import os
import random

import cv2
from PIL import Image

from utils.dataset.tools.enhance_image import gama, test, adaptive, zhifangtu, youxian
from utils.draw_box_utils import draw_objs

classes=['i2', 'i4', 'i5', 'il100', 'il60', 'il80', 'io', 'ip', 'p10', 'p11', 'p12', 'p19', 'p23', 'p26', 'p27', 'p3', 'p5', 'p6', 'pg', 'ph4', 'ph4.5', 'ph5', 'pl100', 'pl120', 'pl20', 'pl30', 'pl40', 'pl5', 'pl50', 'pl60', 'pl70', 'pl80', 'pm20', 'pm30', 'pm55', 'pn', 'pne', 'po', 'pr40', 'w13', 'w32', 'w55', 'w57', 'w59', 'wo']

class_dict={}
for i,cls in enumerate(classes):
    class_dict[i]=cls
category_index = {str(v): str(k) for k, v in class_dict.items()}
# image_dir=r'D:\dataset\glare\GLARE-20221101T123242Z-001\GLARE\Images\data\my_glare_yolo_dataset\val_new\images'
image_dir=r'D:\dataset\data\new_dataset\annotations\mytt100_origion\test\images'


def convert(box, size):
    """
    size:(h,w)
    """
    xmin = (box[1] - box[3] / 2.) * size[1]
    xmax = (box[1] + box[3] / 2.) * size[1]
    ymin = (box[2] - box[4] / 2.) * size[0]
    ymax = (box[2] + box[4] / 2.) * size[0]
    box = (int(xmin), int(ymin), int(xmax), int(ymax))
    print('size: ',(int(xmax)-int(xmin))*(int(ymax)-int(ymin)))
    print('-------------------------')
    return box

img_list=os.listdir(image_dir)

while 1:
    img=input("输入图片名：")
    img=img+'.jpg'
    # img = random.choice(img_list)
    img_path = os.path.join(image_dir, img)
    suffix=os.path.splitext(img_path)[1]
    label_path = img_path.replace('images', 'labels').replace(suffix, '.txt')
    # print('label_path: ',label_path)
    print("path====>",img_path.split('\\')[-1])
    yuantu=cv2.imread(img_path)
    pil_img = cv2.imread(img_path)
    # enhance_img=zhifangtu(pil_img)
    # enhance_img=youxian(pil_img)
    # enhance_img=test(pil_img)
    # enhance_img=gama(enhance_img,0.4) # gama增强
    # enhance_img = cv2.blur(enhance_img, (3, 3)) # 图像平滑
    # enhance_img = adaptive(pil_img) # 图像平滑
    size = pil_img.shape[:2]
    with open(label_path,'r') as f:
        while 1:
            line=f.readline().strip()
            if len(line) !=0:
                cls=line.split(' ')[0]
                cls=class_dict.get(int(cls))
                box = [float(i) for i in line.split(' ')]
                box=convert(box,size)
                pil_img = cv2.rectangle(pil_img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
                cv2.putText(pil_img, cls, (box[0], box[1] - 2), 0, 1, [0, 0, 255], thickness=2, lineType=cv2.LINE_AA)
            else:
                image=cv2.resize(pil_img,(800,800))
                cv2.imshow("images", image)
                # cv2.imshow("zengqiang", enhance_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                break