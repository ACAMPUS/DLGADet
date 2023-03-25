from torch import nn


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class TransformerBlock(nn.Module):
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None

        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)


class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class ConvReduce(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2,  k=3, s=2, p=1, g=1, n=4, e=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(TransformerBlock(c_, c_, 4, 2) for _ in range(n)))
        self.act = nn.SiLU()
        # self.upsample = nn.Upsample(None,scale_factor=2, mode='nearest')
        self.upsample = nn.ConvTranspose2d(c2, c2, k, s, p)

    def pad_tensor(self,input, divide=2):  # divide 为下采样倍数
        height_org, width_org = input.shape[2], input.shape[3]

        if width_org % divide != 0 or height_org % divide != 0:

            width_res = width_org % divide
            height_res = height_org % divide
            if width_res != 0:
                width_div = divide - width_res
                pad_left = int(width_div / 2)
                pad_right = int(width_div - pad_left)
            else:
                pad_left = 0
                pad_right = 0

            if height_res != 0:
                height_div = divide - height_res
                pad_top = int(height_div / 2)
                pad_bottom = int(height_div - pad_top)
            else:
                pad_top = 0
                pad_bottom = 0

            padding = nn.ReflectionPad2d((pad_left, pad_right, pad_top, pad_bottom))
            input = padding(input)
        else:
            pad_left = 0
            pad_right = 0
            pad_top = 0
            pad_bottom = 0

        height, width = input.data.shape[2], input.data.shape[3]
        assert width % divide == 0, 'width cant divided by stride'
        assert height % divide == 0, 'height cant divided by stride'

        return input, pad_left, pad_right, pad_top, pad_bottom

    def forward(self, x):
        # x = self.pad_tensor(x )
        # if isinstance(x,tuple):
        #     x=x[0]
        identy = x
        x1 = self.conv(x)
        x2 = self.bn(x1)
        x3 = self.act(x2)
        x4 = self.m(x3)
        x5 = self.upsample(x4,output_size=identy.size())
        # print("[identiti]-[conv]-[bn]-[act]-[transformer]-[upsample]: ",identy.shape,x1.shape,x2.shape,x3.shape,x4.shape,x5.shape)
        return identy * x5


class AttnBlock(nn.Module):
    def __init__(self, c1, c2, n=1, e=1):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(TransformerBlock(c_, c_, 4, n) for _ in range(n)))
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.m(x))


