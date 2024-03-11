# Referred to https://medium.com/@karuneshu21/how-to-resnet-in-pytorch-9acb01f36cf5

import torch
import torch.nn as nn

class Bottleneck(nn.Module):
    
    def __init__(self, in_channels, inter_channels, stride):
        super(Bottleneck, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels
            
        self.relu = nn.ReLU()
        
        # 3x3 basic block
        self.conv1_3x3 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, 
                                   kernel_size=3, stride=stride, padding=1, bias=False)
        self.BN_1 = nn.BatchNorm2d(self.inter_channels)
        
        # 3x3 basic block
        self.conv2_3x3 = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.inter_channels, 
                                   kernel_size=3, stride=1, padding=1, bias=False)
        self.BN_2 = nn.BatchNorm2d(self.inter_channels)
        
        # Check whether tensor are the same dim
        self.projection = nn.Sequential()
        if self.in_channels != self.inter_channels:
            self.projection =  nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, 
                                              kernel_size=1, stride=stride, padding=0, bias=False),
                                               nn.BatchNorm2d(self.inter_channels))        
        
    def forward(self, x):
        
        identity = x
        
        # 3x3 conv2d
        x = self.conv1_3x3(x)
        x = self.BN_1(x)
        x = self.relu(x)
        
        # 3x3 conv2d
        x = self.conv2_3x3(x)
        x = self.BN_2(x)
        
        # F(x) + x
        x += self.projection(identity)
            
        # Output ReLU
        x = self.relu(x)
        
        return x

class ResNet18(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(ResNet18,self).__init__()
        self.channels_list = [32, 64, 128]
        
        self.layer = nn.Sequential(
            # Conv net (7x7)x16 with batchnorm layer for input layer
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=7, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            # Max pooling layer
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Two basic residual blocks which have 16, 32, and 64 channels respectively
            self.make_layer(16, self.channels_list[0], stride=1),
            self.make_layer(self.channels_list[0], self.channels_list[1], stride=2),
            
            # Average pooling layer
            nn.AdaptiveAvgPool2d((1, 1)),
            
            # A fully connected layer for output layer
            nn.Flatten(),
            nn.Linear(self.channels_list[1], num_classes)
        )
        
    def make_layer(self, in_channels, inter_channels, stride):
        
        # 2 basic residual blocks
        block_num = 2
        
        blocks = [] 
        blocks.append(Bottleneck(in_channels, inter_channels, stride=stride))
        for _ in range(1, block_num):
            blocks.append(Bottleneck(inter_channels, inter_channels, stride=1))

        return nn.Sequential(*blocks)    
    
    def forward(self,x):
        x = self.layer(x)
        return x