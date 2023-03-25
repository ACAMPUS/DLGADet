


class_names_path=r'E:\paper\code\yolov5-6.1\utils\dfg.txt'

if __name__ == '__main__':
    with open(class_names_path,'r') as f:
        list=f.read()
        result=list.split('\n')
        print(result)