import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torchvision.models import resnet18
import seaborn as sns
import math
import torch.nn as nn
import numpy as np

num_epochs = 100
nums = 600
batch_size = 20
n = nums / batch_size

# 定义10分类网络
model = resnet18(num_classes=10)

# optimizer parameter groups 设置了个优化组：权重，偏置，其他参数
pg0, pg1, pg2 = [], [], []
for k, v in model.named_parameters():
    v.requires_grad = True
    if '.bias' in k:
        pg2.append(v)  # biases
    elif '.weight' in k and '.bn' not in k:
        pg1.append(v)  # apply weight decay
    else:
        pg0.append(v)  # all else

optimizer = optim.SGD(pg0, lr=0.01, momentum=0.937, nesterov=True)
# 给optimizer管理的参数组中增加新的组参数，
# 可为该组参数定制lr,momentum,weight_decay 等在finetune 中常用。
optimizer.add_param_group({'params': pg1, 'weight_decay': 0.0005})  # add pg2 (biases)
optimizer.add_param_group({'params': pg2})  # add pg2 (biases)

lf = lambda x: ((1 + math.cos(x * math.pi / num_epochs)) / 2) * (1 - 0.2) + 0.2
scheduler = LambdaLR(
    optimizer=optimizer,
    lr_lambda=lf,  # 传入一个函数或一个以函数为元素列表，作为学习率调整的策略
)

start_epoch = 0
scheduler.last_epoch = start_epoch - 1

lr0, lr1, lr2, epochs = [], [], [], []
optimizer.zero_grad()
for epoch in range(start_epoch, num_epochs):
    for i in range(int(n)):
        # 训练的迭代次数
        ni = i + n * epoch
        # Warmup 热身的迭代次数
        if ni <= 1000:
            xi = [0, 1000]
            for j, x in enumerate(optimizer.param_groups):
                # 一维线性插值
                x['lr'] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, 0.01 * lf(epoch)])
                if 'momentum' in x:
                    x['momentum'] = np.interp(ni, xi, [0.8, 0.937])

    pass  # iter and train here

    # Scheduler 学习率衰减
    lr = [x['lr'] for x in optimizer.param_groups]
    lr0.append(lr[0])
    lr1.append(lr[1])
    lr2.append(lr[2])

    # 学习率更新
    scheduler.step()
    epochs.append(epoch)

plt.figure()
plt.subplot(221)
plt.plot(epochs, lr0, color="r", label='l0')
plt.legend()

plt.subplot(222)
plt.plot(epochs, lr1, color="b", label='l1')
plt.legend()

plt.subplot(223)
plt.plot(epochs, lr2, color="g", label='l2')
plt.legend()

plt.show()
