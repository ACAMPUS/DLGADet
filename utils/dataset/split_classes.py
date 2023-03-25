

def split_classes(path):
    list=[]
    with open(path,'r') as f:
        while 1:
            line=f.readline().strip()
            if len(line) != 0:
                list.append(line)
            else:
                break
    return list

if __name__ == '__main__':
    path=r'D:\dataset\data\my_tt100k\data\classes.txt'
    list=split_classes(path)
    print(f'总共{len(list)}个')
    print(list)


