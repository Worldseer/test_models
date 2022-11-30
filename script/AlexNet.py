"""vgg in pytorch


[1] Karen Simonyan, Andrew Zisserman

    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""
'''VGG11/13/16/19 in Pytorch.'''
#ADAPTED FROM https://github.com/weiaicunzai/pytorch-cifar100
import torch
import torch.nn as nn

class AlexNet(nn.Module):
    super(AlexNet, self).__init__()
    def __init__(self):
        self.net = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
        )
    def forward(self, x):
        return self.net(x)