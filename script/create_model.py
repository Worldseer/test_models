import math
import torch
import torch.nn as nn 
from . import axialnet,resnet,googlenet,vgg,alexnet
#axial26s
backbone_axial26s = axialnet.AxialAttentionNet(axialnet.AxialBlock,[1, 2, 4, 1],in_dim=3, s=0.5, con2d_groups=1, groups=8, image_size=40*self.r)
#axial50s
backbone_axial50s = axialnet.AxialAttentionNet(axialnet.AxialBlock,[3, 4, 6, 3],in_dim=3, s=0.5, con2d_groups=1, groups=8, image_size=40*self.r)
#resnet26
backbone_resnet26 = resnet.ResNet(resnet.Bottleneck, [1, 2, 4, 1])
#resnet50
backbone_resnet50 = resnet.ResNet(resnet.Bottleneck, [3, 4, 6, 3])
#googlenet
backbone_googlenet = googlenet.googlenet()
#vgg
backbone_vgg = vgg.vgg19_bn()
#alexnet
backbone_alexnet = alexnet.AlexNet()

class AxialGO(nn.Module):
    def __init__(self,emb_dim,num_classes,mlp_expand=2):
        super(AxialGO, self).__init__()
        self.emb_1 = nn.Embedding(21,emb_dim,padding_idx=0)
        self.emb_2 = nn.Embedding(42,emb_dim,padding_idx=0)
        self.emb_3 = nn.Embedding(84,emb_dim,padding_idx=0)
        self.r = int(math.sqrt(emb_dim))
        self.pixel_shuffle = nn.PixelShuffle(self.r)
        self.backbone = backbone_axial26s
        self.out = nn.Sequential(
            nn.Linear(1024*1, 1024*mlp_expand),
            nn.BatchNorm1d(1024*mlp_expand),
            nn.ReLU(inplace=True),
            nn.Linear(1024*mlp_expand, num_classes),
            nn.Sigmoid()
        )
        
        
    def forward(self,x):
        x_1 = self.emb_1(x).permute(0,3,1,2)
        x_1 = self.pixel_shuffle(x_1)
        x_2 = self.emb_2(x).permute(0,3,1,2)
        x_2 = self.pixel_shuffle(x_2)
        x_3 = self.emb_3(x).permute(0,3,1,2)
        x_3 = self.pixel_shuffle(x_3)
        x = torch.cat([x_1,x_2,x_3],dim=1)
        x = self.backbone(x)
        x = self.out(x)
        return x