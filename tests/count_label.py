import os
import matplotlib.pyplot as plt


labels_path=r'D:\dataset\data\new_dataset\annotations\mytt100_origion\train\labels'
label_dict={}


def fig(x_list,y_list):
    plt.bar(x_list, y_list)
    for a, b, i in zip(x_list, y_list, range(len(x_list))):  # zip 函数
        plt.text(a, b + 0.01, "%.2f" % y_list[i], ha='center', fontsize=8)  # plt.text 函数
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


def count(labels):
    for label in labels:
        label=os.path.join(labels_path,label)
        with open(label,'r') as f:
            while 1:
                line = f.readline().strip()
                if len(line) != 0:
                    cls=int(line.split(' ')[0])
                    cls=class_dict[cls]
                    if cls not in label_dict:
                        label_dict[cls]=0
                    label_dict[cls]+=1
                else:
                    break

def map_show(path):
    with open(path, 'r') as f:
        while 1:
            line = f.readline().strip()
            if len(line) != 0:
                cls = int(line.split(' ')[0])
                cls = class_dict[cls]
                if cls not in label_dict:
                    label_dict[cls] = 0
                label_dict[cls] += 1
            else:
                break

if __name__ == '__main__':
    labels=os.listdir(labels_path)
    classes = ['i2', 'i4', 'i5', 'il100', 'il60', 'il80', 'io', 'ip', 'p10', 'p11', 'p12', 'p19', 'p23', 'p26', 'p27',
               'p3', 'p5', 'p6', 'pg', 'ph4', 'ph4.5', 'ph5', 'pl100', 'pl120', 'pl20', 'pl30', 'pl40', 'pl5', 'pl50',
               'pl60', 'pl70', 'pl80', 'pm20', 'pm30', 'pm55', 'pn', 'pne', 'po', 'pr40', 'w13', 'w32', 'w55', 'w57',
               'w59', 'wo']

    class_dict = {}
    for i, cls in enumerate(classes):
        class_dict[i] = cls
    count(labels)
    x_list=[]
    y_list=[]
    for k,v in label_dict.items():
        x_list.append(k)
        y_list.append(v)
    print(label_dict)
    fig(x_list, y_list)

