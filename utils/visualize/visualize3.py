import numpy as np
import skimage
import torch
from torchsummary import summary
import torchvision.transforms as transforms
from models.yolo import Model
from utils.general import intersect_dicts
import matplotlib
matplotlib.use('TkAgg')


weights='../../yolov5s.pt'
device='cuda' if torch.cuda.is_available() else 'cpu'
cfg=r'E:\paper\code\yolov5-6.1\models\yolov5s.yaml'
pic_dir=r'E:\paper\code\yolov5-6.1\data\images\bus.jpg'



ckpt = torch.load(weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
model = Model(cfg, ch=3, nc=20).to(device)  # create
# exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # exclude keys
csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
csd = intersect_dicts(csd, model.state_dict())  # intersect
model.load_state_dict(csd, strict=False)  # load
transform = transforms.ToTensor()


class Hook(object):
    def __init__(self):
        self.module_name = []
        self.features_in_hook = []
        self.features_out_hook = []

    def __call__(self, module, fea_in, fea_out):
        print("hooker working", self)
        self.module_name.append(module.__class__)
        self.features_in_hook.append(fea_in)
        self.features_out_hook.append(fea_out)
        return None


def plot_feature(model, idx, inputs):
    hh = Hook()
    model.model[idx].register_forward_hook(hh)

    # forward_model(model, False)
    model.eval()
    _ = model(inputs)
    print(hh.module_name)
    print(hh.features_in_hook[0][0].shape)
    print(hh.features_out_hook[0].shape)

    out1 = hh.features_out_hook[0]

    total_ft = out1.shape[1]
    first_item = out1[0].cpu().clone()

    plt.figure(figsize=(40, 34))

    for ftidex in range(total_ft):
        if ftidex > 99:
            break
        ft = first_item[ftidex]
        plt.subplot(10, 10, ftidex + 1)

        plt.axis('off')
        # plt.imshow(ft[:,:].detach(),cmap='gray')
        plt.imshow(ft[:, :].detach())
        plt.imsave('./images/'+f'{ftidex}.jpg',ft[:, :].detach().numpy())
    plt.show()


from matplotlib import pyplot as plt
import torch

def get_picture(picture_dir, transform):
    '''
    该算法实现了读取图片，并将其类型转化为Tensor
    '''
    img = skimage.io.imread(picture_dir)
    img256 = skimage.transform.resize(img, (640, 640))
    img256 = np.asarray(img256)
    img256 = img256.astype(np.float32)

    return transform(img256)

input=get_picture(pic_dir,transform)
input=input.unsqueeze(0).cuda()
summary(model,input)
plot_feature(model, 4, input)
