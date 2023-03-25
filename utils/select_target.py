import os
import shutil

from tqdm import tqdm

train_path=r'D:\dataset\data\train'
test_path=r'D:\dataset\data\test'
parent_path=r'D:\dataset\data'

train_img_list=os.listdir(train_path)
test_img_list=os.listdir(test_path)




num=0

def divide_TrainValTest(train_img_list, test_img_list,target):
    '''
    创建文件路径
    :param source: 源文件位置
    :param target: 目标文件位置
    '''
    # for i in ['train', 'val', 'test']:
    #     path = target + '/' + i
    #     if not os.path.exists(path):
    #         os.makedirs(path)

    for root, dirs, files in os.walk(source):
        l = tqdm(files, f'source: [{source}]')
        for file in l:
            dir_suffix = root.split('\\')[-1]
            if file in train_img_list:
                continue
                # shutil.copyfile(os.path.join(root, file), os.path.join(target,'train',dir_suffix, file))
                # shutil.copyfile(os.path.join(root.replace('images','labels'), file.replace('jpg','txt')), os.path.join(target,'train','labels', file.replace('jpg','txt')))
            if file in test_img_list:
                # shutil.copyfile(os.path.join(root, file), os.path.join(target,'test',root.split('\\')[-1], file))
                # shutil.copyfile(os.path.join(root.replace('images', 'labels'), file.replace('jpg', 'txt')),
                #                 os.path.join(target,'test', 'labels', file.replace('jpg', 'txt')))
                continue
            else:
                if file.endswith('jpg'):
                    global num
                    num+=1
                    print(file)

source=r'D:\dataset\data\my_tt100k'
target=r'D:\dataset\data\origin_dataset\my_tt100k'
divide_TrainValTest(train_img_list,test_img_list,target)
print(num)

# train=r'D:\dataset\data\my_tt100k\train\labels'
# val=r'D:\dataset\data\my_tt100k\val\labels'
# test=r'D:\dataset\data\my_tt100k\test\labels'
# t=os.listdir(train)
# v=os.listdir(val)
# te=os.listdir(test)
# s = set(v).intersection(set(te))
# print()








