import cv2
import os

classes=['i2', 'i4', 'i5', 'il100', 'il60', 'il80', 'io', 'ip', 'p10', 'p11', 'p12', 'p19', 'p23', 'p26', 'p27', 'p3', 'p5', 'p6', 'pg', 'ph4', 'ph4.5', 'ph5', 'pl100', 'pl120', 'pl20', 'pl30', 'pl40', 'pl5', 'pl50', 'pl60', 'pl70', 'pl80', 'pm20', 'pm30', 'pm55', 'pn', 'pne', 'po', 'pr40', 'w13', 'w32', 'w55', 'w57', 'w59', 'wo']

class_dict={}
for i,cls in enumerate(classes):
    class_dict[i]=cls
category_index = {str(v): str(k) for k, v in class_dict.items()}

def draw_box_in_single_image(image_path, txt_path):
    # 读取图像
    image = cv2.imread(image_path)

    # 读取txt文件信息
    def read_list(txt_path):
        pos = []
        with open(txt_path, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline()  # 整行读取数据
                if not lines:
                    break
                # 将整行数据分割处理，如果分割符是空格，括号里就不用传入参数，如果是逗号， 则传入‘，'字符。
                p_tmp = [float(i) for i in lines.split(' ')]
                pos.append(p_tmp)  # 添加新读取的数据
                # Efield.append(E_tmp)
                pass
        return pos


    # txt转换为box
    def convert(size, box):
        xmin = (box[1]-box[3]/2.)*size[1]
        xmax = (box[1]+box[3]/2.)*size[1]
        ymin = (box[2]-box[4]/2.)*size[0]
        ymax = (box[2]+box[4]/2.)*size[0]
        box = (int(xmin), int(ymin), int(xmax), int(ymax))
        return box

    pos = read_list(txt_path)
    print(pos)
    tl = int((image.shape[0]+image.shape[1])/2)
    lf = max(tl-1,1)
    for i in range(len(pos)):
        label = str(int(pos[i][0]))
        label=class_dict[int(label)]
        print('label is '+label)
        box = convert(image.shape, pos[i])
        image = cv2.rectangle(image,(box[0], box[1]),(box[2],box[3]),(0,0,255),2)
        cv2.putText(image,label,(box[0],box[1]-2), 0, 1, [0,0,255], thickness=2, lineType=cv2.LINE_AA)
        pass

    if pos:
        cv2.imwrite('./images/{}.png'.format(image_path.split('\\')[-1][:-4]), image)
    else:
        print('None')


    print('./images/{}.png'.format(image_path.split('\\')[-1][:-4]))
    cv2.imshow("images", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

image_dir=r'D:\dataset\data\new_dataset\annotations\mytt100_origion\train_new_scale_20\images'
label_dir=r'D:\dataset\data\new_dataset\annotations\mytt100_origion\train_new_scale_20\labels'

img_folder = r'D:\dataset\data\new_dataset\annotations\mytt100_origion\train_new_scale_20\images'
img_list = os.listdir(img_folder)
img_list.sort()

label_folder = r'D:\dataset\data\new_dataset\annotations\mytt100_origion\train_new_scale_20\labels'
label_list = os.listdir(label_folder)
label_list.sort()

for i in range(len(img_list)):
    image_path = img_folder + "\\" + img_list[i]
    txt_path = label_folder + "\\" + label_list[i]
    draw_box_in_single_image(image_path, txt_path)

