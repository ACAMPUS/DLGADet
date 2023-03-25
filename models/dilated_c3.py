import torch
from torch import nn
import torchsummary


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class DilatedConv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True,
                 dilation=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False, dilation=dilation)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class DilatedBottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, dilation=None,act=True):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        if dilation is None:
            dilation = [2, 3]
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = DilatedConv(c1, c_, 3, 1, dilation=dilation[0],p=dilation[0],act=False)
        self.cv2 = DilatedConv(c_, c2, 3, 1, dilation=dilation[1],p=dilation[1],g=g,act=False)
        self.add = shortcut and c1 == c2
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(x + self.cv2(self.cv1(x))) if self.add else self.cv2(self.cv1(x))


class DialtedC3(nn.Module):
    def __init__(self, c1, c2, n=4, shortcut=True, g=1,
                 e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c1 * e)  # hidden channels
        self.cv1 = DilatedConv(c1, c_, 1, 1, dilation=1)
        self.cv2 = DilatedConv(c1, c_, 1, 1, dilation=1)
        self.cv3 = DilatedConv(2 * c_, c2, 1, dilation=1)  # act=FReLU(c2)
        self.m = nn.Sequential(*(DilatedBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(4)))
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


if __name__ == '__main__':
    input = torch.randn(2, 512, 40, 40)
    encoder = DialtedC3(512, 256, True)
    torchsummary.summary(encoder,input)
    # out = encoder(input)
    # print(out.shape)
