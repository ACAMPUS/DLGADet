import torch
import torch.nn as nn


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.LeakyReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class RFBA(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale=1, map_reduce=2):
        super(RFBA, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // map_reduce
        self.branch0 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=stride, relu=True),
            BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=1, relu=True),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=1, stride=stride, relu=True)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=stride, relu=True),
            BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=2, dilation=2, relu=True),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=1, stride=stride, relu=True)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=stride, relu=True),
            BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=True),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=1, stride=stride, relu=True)
        )

        self.branch3 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=stride, relu=True),
            BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=4, dilation=4, relu=True),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=1, stride=stride, relu=True)
        )

    def forward(self, x):
        identy = x
        x0 = self.branch0(x) + identy
        x1 = self.branch1(x) + identy
        x2 = self.branch2(x) + identy
        x3 = self.branch3(x) + identy
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


if __name__ == '__main__':
    input = torch.randn(16, 512, 40, 40)
    attn = RFB(512, 2048)
    out = attn(input)
    print(out.shape)
