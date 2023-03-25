import os
from random import random
import random

import cv2
import numpy as np
import shutil


# 一次读取batch张图，随机选择batch/2张作为bbox源，将源的bbox粘贴到该batch所有的图上
def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)

classes=['i2', 'i4', 'i5', 'il100', 'il60', 'il80', 'io', 'ip', 'p10', 'p11', 'p12', 'p19', 'p23', 'p26', 'p27', 'p3', 'p5', 'p6', 'pg', 'ph4', 'ph4.5', 'ph5', 'pl100', 'pl120', 'pl20', 'pl30', 'pl40', 'pl5', 'pl50', 'pl60', 'pl70', 'pl80', 'pm20', 'pm30', 'pm55', 'pn', 'pne', 'po', 'pr40', 'w13', 'w32', 'w55', 'w57', 'w59', 'wo']

class_dict={}
for i,cls in enumerate(classes):
    class_dict[i]=cls


def load_img_from_train_v2(img_list,k=6):
    """
        从train中随机选择batch张图
        """
    # 被选中的
    try:
        imgs_list = random.sample(img_list, k)
    except Exception as e:
        img_list_temp = os.listdir(train_img_path)
        temp=k-len(img_list)
        img_list_temp1 = random.sample(img_list_temp, temp)
        imgs_list=img_list+img_list_temp1
    # 被选中的
    annotations = []
    img_path_list = []
    image_index_dict = {}
    for i in range(len(imgs_list)):
        img_path = os.path.join(train_img_path, imgs_list[i])
        img_path_list.append(img_path)
        suffix = os.path.splitext(img_path)[1]
        label_path = img_path.replace('images', 'labels').replace(suffix, '.txt')
        with open(label_path, 'r') as f:
            txt = f.readlines()
            image_index_dict[i] = txt
            annotations.extend([line.strip() for line in txt if len(line.strip().split()[1:]) != 0])
    leave_list = list(set(img_list) - set([imgs_list[0]]))
    if len(annotations)!=len(set(annotations)):
        print("============")
    return leave_list, annotations,  img_path_list, image_index_dict




# def load_img_from_train_v2(img_list,k=5):
#     """
#         从train中随机选择batch张图
#         """
#     # 被选中的
#     img = random.choice(img_list)
#     imgs_list = random.choices(img_list, k=k)
#     all_list=[img]+imgs_list
#     # 被选中的
#     annotations = []
#     img_path_list = []
#     image_index_dict = {}
#     for i in range(len(all_list)):
#         img_path = os.path.join(train_img_path, all_list[i])
#         img_path_list.append(img_path)
#         suffix = os.path.splitext(img_path)[1]
#         label_path = img_path.replace('images', 'labels').replace(suffix, '.txt')
#         with open(label_path, 'r') as f:
#             txt = f.readlines()
#             image_index_dict[i] = txt
#             annotations.extend([line.strip() for line in txt if len(line.strip().split()[1:]) != 0])
#     leave_list = list(set(img_list) - set([img]))
#     return leave_list, annotations,  img_path_list, image_index_dict

def load_img_from_train(train_img_path,img_list,batch=20):
    """
    从train中随机选择batch张图
    """
    # 被选中的
    annotations=[]
    select_img_list=[]
    img_path_list=[]
    image_index_dict = {}
    for i in range(batch):
        img = random.choice(img_list)
        select_img_list.append(img)
        img_path = os.path.join(train_img_path, img)
        img_path_list.append(img_path)
        suffix = os.path.splitext(img_path)[1]
        label_path = img_path.replace('images', 'labels').replace(suffix, '.txt')
        with open(label_path, 'r') as f:
            txt = f.readlines()
            image_index_dict[i]=txt
            txt_sample=random.choice(txt)
            select_txt_sample_list=[]
            select_txt_sample_list.append(txt_sample)
            annotations.extend([line.strip() for line in select_txt_sample_list if len(line.strip().split()[1:]) != 0])
    leave_list=list(set(img_list)-set(select_img_list))
    return leave_list,annotations,select_img_list,img_path_list,image_index_dict


def add_patch_in_img(new_labels, ls, image,img_src,l_h,l_w):
    ls = ls.astype(np.int)
    # image[new_label[2]:new_label[4], new_label[1]:new_label[3], :] = image[l[2]:l[4], l[1]:l[3], :]
    i=0
    for l,new_label in zip(ls,new_labels):
        origin_label_img=img_src[l[2]:l[4], l[1]:l[3], :]
        new_label_img=cv2.resize(origin_label_img,(l_w[i],l_h[i]))
        image[new_label[2]:new_label[4], new_label[1]:new_label[3], :] = new_label_img
        i+=1
        # cv2.rectangle(image, (int(new_label[1]), int(new_label[2])), (int(new_label[3]), int(new_label[4])),
        #               (0, 0, 255), 2)
        # # image = cv2.resize(image, (880, 880))
        # cv2.imshow('a', image)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
    return image

def sample_half(selected_list):
    random_list=random.sample(selected_list, len(selected_list)/2)
    label_idx = [selected_list[idx] for idx in random_list]


def transform_annotations(annotations):
    label_list=[]
    for line in annotations:
        label_list.append([float(label) for label in line.split(' ')])
    return label_list


def transform_annotations_dict(annotations):
    annotations_dict={}
    for i,vals in enumerate(annotations.values()):
        label_list = []
        for line in vals:
            label_list.append(np.array([float(label) for label in line.split(' ')]))
        annotations_dict[i]=label_list
    return annotations_dict

def compute_overlap(annot_a, annot_b):
    # todo:粘贴上的标签和原始标签的重叠问题，该值可以作为超参数提供一个重叠的iou阈值
    if annot_a is None: return False
    left_max = max(annot_a[1], annot_b[1])
    top_max = max(annot_a[2], annot_b[2])
    right_min = min(annot_a[3], annot_b[3])
    bottom_min = min(annot_a[4], annot_b[4])
    inter = max(0, (right_min - left_max)) * max(0, (bottom_min - top_max))
    if inter != 0:
        return True
    else:
        return False

def donot_overlap(new_l, labels):
    for l in labels:
        for new in new_l:
            if compute_overlap(new, l): return False
    return True

def create_copy_label(h, w, l, labels,s):
    """
    h:是原图的一半
    w:是原图的一半
    l:待拷贝的bbox标签
    labels:改图本身的labels
    """
    l = l.astype(np.int)
    # bbox的宽高
    l_h, l_w = ((l[:,4] - l[:,2])*s).astype(np.int), ((l[:,3] - l[:,1])*s).astype(np.int)
    for epoch in range(30):
        # l_w/2,box的一半，w是整幅图的一半，把粘贴的中心点坐标限制在了1/4图像切不会超过1/4图像宽高的空间里面
        random_x, random_y = np.random.randint((l_w / 2).astype(np.int), (w - l_w / 2).astype(np.int)), \
                             np.random.randint((l_h / 2).astype(np.int), (h - l_h / 2).astype(np.int))
        xmin, ymin = random_x - l_w / 2, random_y - l_h / 2
        # xmax, ymax = xmin + l_w, ymin + l_h
        xmax, ymax = random_x + l_w / 2, random_y + l_h / 2
        if np.any(xmin < 0) or np.any(xmax > w) or np.any(ymin < 0) or np.any(ymax > h):
            continue
        new_l = np.vstack((l[:,0],xmin,ymin,xmax,ymax)).T.astype(np.int)
        if np.any((new_l[:,4] - new_l[:,2])!=l_h) or np.any((new_l[:,3] - new_l[:,1])!=l_w):
            continue
        # 有交集 返回false，无交集，返回true.这里的设置有交集直接下一轮循环
        if donot_overlap(new_l, labels) is False:
            continue
        return new_l
    return None


def convert(size, box):
    dw = 1. / (size[:,0])
    dh = 1. / (size[:,1])
    x = (box[:,0] + box[:,2]) / 2.0
    y = (box[:,1] + box[:,3]) / 2.0
    w = box[:,2]-box[:,0]
    h = box[:,3]-box[:,1]
    # round函数确定(xmin, ymin, xmax, ymax)的小数位数
    x = np.around(x * dw, 6)
    w = np.around(w * dw, 6)
    y = np.around(y * dh, 6)
    h = np.around(h * dh, 6)
    if np.sum(x>=1) or np.sum(y>=1) or np.sum(w>=1) or np.sum(h>=1):
        print('x:',x)
        print('y:',y)
        print('w:',w)
        print('h:',h)
    return np.stack((x,y,w,h),axis=1)


def xywh2xyxy(x, size):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    dw = size[0]
    dh = size[1]
    y = np.copy(x)
    y[:, 0] = np.around((x[:, 0] - x[:, 2] / 2) * dw)  # top left x
    y[:, 1] = np.around((x[:, 1] - x[:, 3] / 2) * dh)  # top left y
    y[:, 2] = np.around((x[:, 0] + x[:, 2] / 2) * dw)  # bottom right x
    y[:, 3] = np.around((x[:, 1] + x[:, 3] / 2) * dh)  # bottom right y
    # if y.min()<0:
    #     print('xxxxxxxxxxxx')
    # 78961.jpg边缘图片出现这种情况。xcenter-w/2<0,令其为0
    y=np.maximum(y,0)
    return y



init_seeds()
flag=''
# train_img_path=r'D:\dataset\glare\GLARE-20221101T123242Z-001\GLARE\Images\data\my_glare_yolo_dataset\val\images'
# train_img_path=r'D:\dataset\data\new_dataset\annotations\mytt100_origion\test\images'
train_img_path=r'/root/autodl-tmp/datasets/tt100k_2021/mytt100/train_old/images'
suffix = '.jpg' # todo:修改图片格式
img_list=os.listdir(train_img_path)
result = []
# 0.5-1.5
# for j in range(5, 16):
#     result.append(j / 10)
# 0.8-1.2
for j in range(8, 13):
    result.append(j / 10)


while len(img_list)!=0:
    leave_list, annotations,  img_path_list, image_index_dict=load_img_from_train_v2(img_list)
    select_main_labels=transform_annotations_dict(image_index_dict)
    select_main_imgs_list=[]
    five_labels=[]
    labels_dict={}
    five_imgs=[]
    for img_path,select_main_label in zip(img_path_list,select_main_labels):

        img = cv2.imread(img_path)
        # if select_main_label[1]==0.283951:
        #     print()select_main_labels[select_main_label]
        labels=np.array(select_main_labels[select_main_label])
        size = img.T.shape[-2:]
        bbox = labels[:, 1:]
        bbox = xywh2xyxy(bbox, size)
        labels[:, 1:] = bbox
        labels_dict[select_main_label]=labels
        five_labels.append(labels)
        five_imgs.append(img)
    # 为第一张图粘贴five_labels中的图
    j=0
    img=five_imgs[0]
    h, w = img.shape[0], img.shape[1]
    wh=np.array([w,h])
    src_labels=five_labels[0]
    # if h==540 and w==810:
    #     print()
    for i in range(1,len(five_labels)):
        s=random.choice(result)
        # s=1
        l=five_labels[i]
        l_h, l_w = ((l[:,4] - l[:,2])*s).astype(np.int), ((l[:,3] - l[:,1])*s).astype(np.int)
        new_label=create_copy_label(h, w, l, l,s)
        if np.any((new_label[:,4] - new_label[:,2]) != l_h) or np.any((new_label[:,3] - new_label[:,1]) != l_w):
            continue
        # cv2.rectangle(img, (int(labels[i][1]), int(labels[i][2])), (int(labels[i][3]), int(labels[i][4])),
        #               (0, 0, 255), 2)
        # cv2.imshow('a', img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        if new_label is not None:
            src_labels1 = np.append(src_labels, new_label, axis=0)
            _,tt=np.unique(src_labels1,axis=0,return_index=True)
            if len(tt)<len(src_labels1):
                print("=====duplicate labels=====")
                continue
            src_labels = np.append(src_labels, new_label, axis=0)
            # new_label是一个位置，把需要拷贝的l放到new_label的位置
            try:
                img = add_patch_in_img(new_label, l, img,five_imgs[i],l_h,l_w)
            except Exception as e:
                print(e)
                continue

            # cv2.rectangle(img, (int(labels[i+1][1]), int(labels[i+1][2])), (int(labels[i+1][3]), int(labels[i+1][4])),
            #               (0, 0, 255), 2)
            # image = cv2.resize(img, (880, 880))
            # cv2.imshow('a', image)
            # cv2.waitKey()
            # cv2.destroyAllWindows()

            wh1=np.array([img.shape[1],img.shape[0]])
            wh=np.vstack((wh,wh1))
    labels=src_labels
    bbox = labels[:, 1:]
    # size=wh[1:]
    size=np.tile(np.array([2048,2048]),(bbox.shape[0],1))
    bbox = convert(size,bbox)
    labels[:, 1:] = bbox
    # todo:修改数据文件夹
    # todo:修改数据文件夹
    target_path = img_path_list[j].replace('train_old', 'train_select_5_no_duplicate').replace('images', 'labels').replace(suffix,
                                                                                                              '.txt')
    src_path = img_path_list[j].replace('images', 'labels').replace(suffix, '.txt')
    # shutil.copy(src_path, target_path)
    f = open(target_path, "w")
    for l in labels:
        # np.savetxt(f, l, fmt=' '.join(['%i'] + ['%1.6f']*4))
        np.savetxt(f, l.reshape(1, 5), fmt=' '.join(['%i'] + ['%1.6f'] * 4))
    f.close()
    # np.savetxt(img_path_list[j].replace('train','train_new').replace('images','labels').replace(suffix,'.txt'),labels,fmt=' '.join(['%i'] + ['%1.6f']*4))
    cv2.imwrite(img_path_list[j].replace('train_old', 'train_select_5_no_duplicate'), img, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

    image_path = img_path_list[j].replace('train_old', 'train_select_5_no_duplicate')
    label_path = img_path_list[j].replace('train_old', 'train_select_5_no_duplicate').replace('images', 'labels').replace(suffix,
                                                                                                             '.txt')
    def convert1(box, size):
        """
        size:(h,w)
        """
        xmin = (box[1] - box[3] / 2.) * size[1]
        xmax = (box[1] + box[3] / 2.) * size[1]
        ymin = (box[2] - box[4] / 2.) * size[0]
        ymax = (box[2] + box[4] / 2.) * size[0]
        box = (int(xmin), int(ymin), int(xmax), int(ymax))
        return box



    yuantu = cv2.imread(image_path)
    size = yuantu.shape[:2]
    with open(label_path, 'r') as f:
        while 1:
            line = f.readline().strip()
            if len(line) != 0:
                cls = line.split(' ')[0]
                cls=class_dict.get(int(cls))
                box = [float(i) for i in line.split(' ')]
                box = convert1(box, size)
                yuantu = cv2.rectangle(yuantu, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
                # cv2.putText(yuantu, cls, (box[0], box[1] - 2), 0, 1, [68, 255, 255], thickness=2,
                #             lineType=cv2.LINE_AA)
                # image = cv2.resize(yuantu, (880, 880))
                # cv2.imshow("images", image)
                # # cv2.imshow("zengqiang", enhance_img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
            else:
                # image=cv2.resize(yuantu,(880,880))
                # cv2.imshow("images", image)
                # # cv2.imshow("zengqiang", enhance_img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                break

    j+=1
    img_list = leave_list
