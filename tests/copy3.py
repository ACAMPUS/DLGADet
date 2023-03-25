import os

import numpy as np
import random
import cv2


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)


def xyxy_xywh(size, bbox):
    """
    将bbox的左上角点、右下角点坐标的格式，转换为bbox中心点 + bbox的w,h的格式，并进行归一化
    size: [weight, height]
    bbox: [Xmin, Ymin, Xmax, Ymax]
    即：xyxy（左上右下） ——> xywh（中心宽高）
    xyxy（左上右下）:左上角的xy坐标和右下角的xy坐标
    xywh（中心宽高）:边界框中心点的xy坐标和图片的宽度和高度
    """
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (bbox[0] + bbox[2]) / 2.0
    y = (bbox[1] + bbox[3]) / 2.0
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = x * dw
    y = y * dh
    w = w * dw
    h = h * dh
    return (x, y, w, h)


def xywh_xyxy(size, bbox):
    dw = size[0]
    dh = size[1]
    w = bbox[2] * dw
    h = bbox[3] * dh
    xc = bbox[0] * dw
    yc = bbox[1] * dh
    xmin = xc - w / 2
    ymin = yc - h / 2
    ymax = yc + h / 2
    xmax = xc + w / 2
    return xmin, ymin, xmax, ymax


def xywh2xyxy(x, size):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    dw = size[0]
    dh = size[1]
    y = np.copy(x)
    y[:, 0] = np.around((x[:, 0] - x[:, 2] / 2) * dw)  # top left x
    y[:, 1] = np.around((x[:, 1] - x[:, 3] / 2) * dh)  # top left y
    y[:, 2] = np.around((x[:, 0] + x[:, 2] / 2) * dw)  # bottom right x
    y[:, 3] = np.around((x[:, 1] + x[:, 3] / 2) * dh)  # bottom right y
    return y


class copy_paste(object):
    def __init__(self, thresh=50 * 80, prob=0.5, copy_times=3, epochs=30, all_objects=False, one_object=False):
        self.thresh = thresh
        self.prob = prob,
        self.copy_time = copy_times
        self.epochs = epochs
        self.all_object = all_objects
        self.one_object = one_object

    def issmallobject(self, h, w):
        if h * w <= self.thresh:
            return True
        else:
            return False

    def compute_overlap(self, annot_a, annot_b):
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

    def donot_overlap(self, new_l, labels):
        for l in labels:
            if self.compute_overlap(new_l, l): return False
        return True

    def create_copy_label(self, h, w, l, labels):
        """
        h:是原图的一半
        w:是原图的一半
        l:待拷贝的bbox标签
        labels:改图本身的labels
        """
        l = l.astype(np.int)
        # bbox的宽高
        l_h, l_w = l[4] - l[2], l[3] - l[1]
        for epoch in range(self.epochs):
            # l_w/2,box的一半，w是整幅图的一半，把粘贴的中心点坐标限制在了1/4图像切不会超过1/4图像宽高的空间里面
            random_x, random_y = np.random.randint(int(l_w / 2), int(w - l_w / 2)), \
                                 np.random.randint(int(l_h / 2), int(h - l_h / 2))
            xmin, ymin = random_x - l_w / 2, random_y - l_h / 2
            xmax, ymax = xmin + l_w, ymin + l_h
            if xmin < 0 or xmax > w or ymin < 0 or ymax > h:
                continue
            new_l = np.array([l[0], xmin, ymin, xmax, ymax]).astype(np.int)
            # 有交集 返回false，无交集，返回true.这里的设置有交集直接下一轮循环
            if self.donot_overlap(new_l, labels) is False:
                continue
            return new_l
        return None

    def add_patch_in_img(self, new_label, l, image):
        l = l.astype(np.int)
        image[new_label[2]:new_label[4], new_label[1]:new_label[3], :] = image[l[2]:l[4], l[1]:l[3], :]
        return image

    def __call__(self, image, labels):
        """
        image: numpy.ndarry (1280, 1280, 3)
        labels: [:, class+xyxy] 没用归一化的  numpy.ndarry (6, 5)
        """
        # h, w = image.shape[0] / 2, image.shape[1] / 2
        h, w = image.shape[0] , image.shape[1]
        small_object_list = []
        for i in range(labels.shape[0]):
            label = labels[i]
            label_h, label_w = label[4] - label[2], label[3] - label[1]
            if self.issmallobject(label_h, label_w):
                small_object_list.append(i)
        l = len(small_object_list)
        if l == 0:
            return image, labels

        # 随机copy的个数
        copy_object_num = np.random.randint(0, l)
        # 复制所有
        if self.all_object:
            copy_object_num = l
        if self.one_object:
            copy_object_num = 1

        # 在 0~l-1 之间随机取copy_object_num个数
        random_list = random.sample(range(l), copy_object_num)
        label_idx_of_small_object = [small_object_list[idx] for idx in random_list]
        select_label = labels[label_idx_of_small_object, :]
        select_label=labels #todo: 硬编码改回去

        # for idx in range(copy_object_num):  #todo: 硬编码改回去
        for idx in range(1):  #todo: 硬编码改回去
            l = select_label[idx]
            l_h, l_w = l[4] - l[2], l[3] - l[1]
            if self.issmallobject(l_h, l_w) is False:
                continue

            for i in range(self.copy_time):
                new_label = self.create_copy_label(h, w, l, labels)
                if new_label is not None:
                    # new_label是一个位置，把需要拷贝的l放到new_label的位置
                    image = self.add_patch_in_img(new_label, l, image)
                    # cv2.imshow('a',image)
                    # cv2.waitKey()
                    # cv2.destroyAllWindows()
                    labels = np.append(labels, new_label.reshape(1, -1), axis=0)

        return image, labels

init_seeds(seed=0)
# label_path = r'D:\dataset\glare\GLARE-20221101T123242Z-001\GLARE\Images\data\my_glare_yolo_dataset\train\labels\2021_0818_115908_062B-doNotEnter-start9185.MOV_image0.txt'
label_path = r'D:\dataset\glare\GLARE-20221101T123242Z-001\GLARE\Images\data\my_glare_yolo_dataset\train\labels\2021_0818_115908_062B-doNotEnter-start9185.MOV_image0.txt'
img_path = label_path.replace('labels', 'images').replace('.txt', '.png')
with open(label_path, 'r') as f:
    txt = f.readlines()
    annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
label_list = []
for line in annotations:
    label_list = [float(label) for label in line.split(' ')]

labels = np.array(label_list).reshape(-1, 5)

img = cv2.imread(img_path)
size = img.T.shape[-2:]
bbox=labels[:,1:]
bbox=xywh2xyxy(bbox,size)
labels[:,1:]=bbox
image, labels = copy_paste().__call__(img, labels)
# print(a)
cv2.imshow('a', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(labels)
