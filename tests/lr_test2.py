import torch
import torchvision
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
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 6], gamma=0.1)
epochs=9
for epoch in range(epochs):
    optimizer.step()
    scheduler.step()
    print('get_lr[epoch%d]: '%epoch,scheduler.get_lr())
    print('get_last_lr[epoch%d]: '%epoch,scheduler.get_last_lr())
    # if epoch !=epochs-1:
    #     print(epoch, scheduler.get_lr())
    # else:
    #     print(epoch,scheduler.get_last_lr())
