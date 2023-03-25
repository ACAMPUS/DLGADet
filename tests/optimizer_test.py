import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import itertools


initial_lr = 0.2

class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)

    def forward(self, x):
        pass

net_1 = model()
net_2 = model()

optimizer_1 = torch.optim.Adam(net_1.parameters(), lr = initial_lr)
print("******************optimizer_1*********************")
print("optimizer_1.defaults：", optimizer_1.defaults)
print("optimizer_1.param_groups长度：", len(optimizer_1.param_groups))
print("optimizer_1.param_groups一个元素包含的键：", optimizer_1.param_groups[0].keys())
print()
####################################################################################


optimizer_2 = torch.optim.Adam([*net_1.parameters(), *net_2.parameters()], lr = initial_lr)
# optimizer_2 = torch.opotim.Adam(itertools.chain(net_1.parameters(), net_2.parameters())) # 和上一行作用相同
print("******************optimizer_2*********************")
print("optimizer_2.defaults：", optimizer_2.defaults)
print("optimizer_2.param_groups长度：", len(optimizer_2.param_groups))
print("optimizer_2.param_groups一个元素包含的键：", optimizer_2.param_groups[0].keys())
print()
####################################################################################


optimizer_3 = torch.optim.Adam([{"params": net_1.parameters()}, {"params": net_2.parameters()}], lr = initial_lr)
print("******************optimizer_3*********************")
print("optimizer_3.defaults：", optimizer_3.defaults)
print("optimizer_3.param_groups长度：", len(optimizer_3.param_groups))
print("optimizer_3.param_groups一个元素包含的键：", optimizer_3.param_groups[0].keys())
####################################################################################
# ******************optimizer_3*********************

