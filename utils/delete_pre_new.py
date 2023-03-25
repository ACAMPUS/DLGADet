# 列出所有文件，如果有比当前更大的，删除以前的文件
import argparse
import datetime
import os
import time

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('folder_path', type=str, default='', help='delete folder path')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

def remove_files(opt):
    # 获取文件夹下所有文件
    folder_path=opt.folder_path
    files = os.listdir(folder_path)
    # 过滤出以"best_"开头的文件
    pt_files = [f for f in files if f.startswith("best_") and f.endswith(".pt")]
    # 过滤出以"epoch_"开头的文件
    epoch_pt_files = [f for f in files if f.startswith("epoch_") and f.endswith(".pt")]
    # 按照编号排序
    pt_files.sort(key=lambda x: int(x[5:-3]))
    epoch_pt_files.sort(key=lambda x: int(x[6:-3]))
    # # 删除编号较小的文件
    for f in pt_files[:-3]:
        os.remove(os.path.join(folder_path, f))
        print("delete best file %s at [%s]"%(f,datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    # 删除编号较小epoch的文件
    for f in epoch_pt_files[:-3]:
        os.remove(os.path.join(folder_path, f))
        print("delete epoch file %s at [%s]" % (f, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))


# 使用示例
# 执行每小时的检查和删除任务
if __name__ == '__main__':
    opt = parse_opt()
    while True:
        remove_files(opt)
        time.sleep(5)  # 每小时休眠一次

