import mmcv
import os
import sys
import json
import cv2
import numpy as np
from tqdm import tqdm


def read_file_to_list(classes_txt_path):
    """
    读取classes.txt,返回list列表与其长度
    """
    with open(classes_txt_path, 'r') as f:
        list = []
        while 1:
            line = f.readline().strip()
            if len(line) != 0:
                list.append(line)
            else:
                break
    return list, len(list)


def load_train_or_val_csv_info(train_or_val_csv_info_path):
    img_info_list = mmcv.list_from_file(train_or_val_csv_info_path)
    return img_info_list


def to_coco(categories_list, dataset_flag, dataset_root, train_or_val_img_txt_relative_save_path):
    assert dataset_flag in ['train', 'val',
                            'test'], f"dataset_flag: {dataset_flag} does not support,support list ['train','val','test']"
    train_dataset = {'info': {}, 'licenses': [], 'categories': [], 'images': [], 'annotations': []}
    val_dataset = {'info': {}, 'licenses': [], 'categories': [], 'images': [], 'annotations': []}
    label = {}  # 记录每个标志类别的id
    count = {}  # 记录每个类别的图片数
    owntype_sum = {}
    info = {
        "year": 2022,  # 年份
        "version": '1.0',  # 版本
        "description": "glare",  # 数据集描述
        "contributor": "glare",  # 提供者
        "url": 'glare',  # 下载地址
        "date_created": 2022 - 10
    }
    licenses = {
        "id": 1,
        "name": "null",
        "url": "null",
    }
    train_dataset['info'] = info
    val_dataset['info'] = info
    train_dataset['licenses'] = licenses
    val_dataset['licenses'] = licenses

    # 建立类别和id的关系
    for i, cls in enumerate(categories_list):
        train_dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'traffic_sign'})
        val_dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'traffic_sign'})
        label[cls] = i
        count[cls] = 0
        owntype_sum[cls] = 0

    img_info_list = load_train_or_val_csv_info(train_or_val_csv_info_path)
    img_info_list_length = len(img_info_list) - 1
    print(f'{dataset_flag}_img_length: ', img_info_list_length)
    if dataset_flag == 'train':
        dataset = train_dataset
    elif dataset_flag == 'val':
        dataset = val_dataset
    obj_id = 1
    train_data_path = []
    pbar = tqdm(img_info_list[1:], desc=f'generate_coco_{dataset_flag}_json: ')  # 跳过csv header
    for image_info in pbar:  # 跳过csv header
        line_list = image_info.split(',')
        file_path = line_list[0]
        train_data_path.append(file_path)
        image_name = file_path.split('/')[-1]
        cls = line_list[1]
        xmin = float(line_list[2])
        ymin = float(line_list[3])
        xmax = float(line_list[4])
        ymax = float(line_list[5])
        width = float(xmax - xmin)
        height = float(ymax - ymin)
        dataset['annotations'].append({
            'area': width * height,
            'bbox': [xmin, ymin, width, height],
            'category_id': label[cls],
            'id': obj_id,
            'iscrowd': 0,
        })
        obj_id += 1
        img_path = os.path.join(dataset_root, file_path)
        assert os.path.exists(img_path), f"img_file: {img_path} does not exist"
        # 用opencv读取图片，得到图像的宽和高
        im = cv2.imread(img_path)
        H, W, _ = im.shape
        # 添加图像的信息到dataset中
        dataset['images'].append({'file_name': image_name,
                                  'width': W,
                                  'height': H})
    # 保存结果
    with open(train_or_val_img_txt_relative_save_path, 'a', encoding='utf-8') as f:
        for path in train_data_path:
            f.write(path + '\n')
    for phase in ['train', 'val', 'test']:
        json_name = os.path.join(dataset_root, 'data/{}.json'.format(phase))
        if phase == 'train' and dataset_flag == 'train':
            with open(json_name, 'w', encoding="utf-8") as f:
                json.dump(train_dataset, f, ensure_ascii=False, indent=1)
        if phase == 'val' and dataset_flag == 'val':
            with open(json_name, 'w', encoding="utf-8") as f:
                json.dump(val_dataset, f, ensure_ascii=False, indent=1)
    return img_info_list_length


def write_dataset_info(dataset_info_path, data_info):
    with open(dataset_info_path, 'a') as f:
        json.dump(data_info, f, ensure_ascii=False, indent=1)
    pass


# 文件路径必须为左斜杠
classes_txt_path = r'D:/dataset/glare/GLARE-20221101T123242Z-001/GLARE/Images/categories.txt'  
train_or_val_csv_info_path = r'D:/dataset/glare/GLARE-20221101T123242Z-001/GLARE/Images/val.csv'
dataset_root = r'D:/dataset/glare/GLARE-20221101T123242Z-001/GLARE/Images'
train_or_val_img_txt_relative_save_path = 'data/my_val_data.txt' # 格式必须一样，如换成'data/my_val_data.txt'
dataset_info_path = './data/my_dataset_info.txt'
assert os.path.exists('./data'), "data dir does not exist"

class_names, all_dataset_class_length = read_file_to_list(classes_txt_path)
print('classnames length: ', all_dataset_class_length)

dataset_flag_selector = lambda x: x.split('_')[1]
dataset_flag=dataset_flag_selector(train_or_val_img_txt_relative_save_path)
assert train_or_val_csv_info_path.split('/')[-1].split('.')[0] in ['train','val','test'], 'support [train,val,test] only'
assert os.path.exists(train_or_val_csv_info_path), "train_or_val_csv_info_path not exist..."
assert os.path.exists(dataset_root), "dataset_root path not exist..."
dataset_info_length = to_coco(class_names, dataset_flag, dataset_root, train_or_val_img_txt_relative_save_path)
dataset_info_dict = {
    'dataset_type': dataset_flag,
    'dataset_length': dataset_info_length,
    'all_dataset_class_length': all_dataset_class_length,
    f'{dataset_flag}_img_txt_relative_path': train_or_val_img_txt_relative_save_path,
}
write_dataset_info(dataset_info_path, dataset_info_dict)
