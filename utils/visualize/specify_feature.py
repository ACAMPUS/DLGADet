import torch
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor


backbone=torchvision.models.efficientnet_b5()
print(backbone)
backbone=create_feature_extractor(backbone,return_nodes={"features.6":"0"})
out=backbone(torch.randn(1, 3, 224, 224)) # out_channels看这个的输出channels
print(out['0'].shape)
backbone.out_channels=512
