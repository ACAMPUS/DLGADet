import torch
import torchvision
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from torch.nn import Parameter

learing_rate = 0.1
# model = torchvision.models.resnet18()
# optimizer = torch.optim.SGD(model.parameters(), lr=learing_rate,
#                             momentum=0.9,
#                             weight_decay=5e-5)


model = [Parameter(torch.randn(2, 2, requires_grad=True))]
optimizer = torch.optim.SGD(model, lr=learing_rate,
                            momentum=0.9,
                            weight_decay=5e-5)
lmbda =lambda epoch:0.95
scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer,  lr_lambda=lmbda)

epochs=100
lr_list, epoch_list=[], []
for epoch in range(epochs):
    # Scheduler 学习率衰减
    lr = [x['lr'] for x in optimizer.param_groups]
    lr_list.append(lr[0])
    # print('get_lr[epoch%d]: '%epoch,scheduler.get_lr())
    print('get_last_lr[epoch%d]: ' % epoch, scheduler.get_last_lr())
    epoch_list.append(epoch)
    optimizer.step()
    scheduler.step()

    

plt.plot(epoch_list, lr_list, color="r", label='l0')
plt.show()

