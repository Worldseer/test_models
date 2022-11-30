import math
import torch
import torch.nn as nn 
from . import axialnet,resnet,googlenet,vgg,alexnet



class AxialGO(nn.Module):
    def __init__(self,emb_dim,winding_size,num_classes,mlp_expand=2):
        super(AxialGO, self).__init__()
        self.emb_1 = nn.Embedding(21,emb_dim,padding_idx=0)
        self.emb_2 = nn.Embedding(42,emb_dim,padding_idx=0)
        self.emb_3 = nn.Embedding(84,emb_dim,padding_idx=0)
        self.r = int(math.sqrt(emb_dim))
        self.pixel_shuffle = nn.PixelShuffle(self.r)
        #axial26s
        self.backbone = axialnet.AxialAttentionNet(axialnet.AxialBlock,[1, 2, 4, 1],in_dim=3, s=0.5, con2d_groups=1, groups=8, image_size=winding_size*self.r)
        #axial50s
        #self.backbone = axialnet.AxialAttentionNet(axialnet.AxialBlock,[3, 4, 6, 3],in_dim=3, s=0.5, con2d_groups=1, groups=8, image_size=winding_size*self.r)
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
        
#The dimensionality of Linear's input needs to vary 
#according to the different backbone networks
class OtherBackBone(nn.Module):
    def __init__(self,emb_dim,num_classes,mlp_expand=2):
        super(OtherBackBone, self).__init__()
        self.emb_1 = nn.Embedding(21,emb_dim,padding_idx=0)
        self.emb_2 = nn.Embedding(42,emb_dim,padding_idx=0)
        self.emb_3 = nn.Embedding(84,emb_dim,padding_idx=0)
        self.r = int(math.sqrt(emb_dim))
        self.pixel_shuffle = nn.PixelShuffle(self.r) 
        backbone = resnet.ResNet(resnet.Bottleneck, [1, 2, 4, 1]) #resnet26 linear_out_dim=2048
        
        #backbone = resnet.ResNet(resnet.Bottleneck, [3, 4, 6, 3]) #resnet50    linear_out_dim=2048
        
        #backbone = googlenet.googlenet() #googlenet    linear_out_dim=1024
        
        #backbone = vgg.vgg19_bn()  #vgg    linear_out_dim=512
    
        #backbone = alexnet.AlexNet()   #alexnet    linear_out_dim=2304
        
        self.out = nn.Sequential(#
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