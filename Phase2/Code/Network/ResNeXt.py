import torch
import torch.nn as nn

class BottleneckBlock(nn.Module):
    def __init__(self, in_channel, out_width, C, stride, expansion):
        super(BottleneckBlock, self).__init__()
        self.in_channel = in_channel
        self.inter_channel = out_width * C
        self.C = C
        self.expansion = expansion
        
        
        self.conv1x1_1 = nn.Conv2d(self.in_channel, self.inter_channel, kernel_size=1, bias=False)
        self.BN_1 = nn.BatchNorm2d(self.inter_channel)
        
        self.conv3x3 = nn.Conv2d(self.inter_channel, self.inter_channel, kernel_size=3, stride=stride, padding=1, groups=C, bias=False)
        self.BN_2 = nn.BatchNorm2d(self.inter_channel)
        
        self.conv1x1_2 = nn.Conv2d(self.inter_channel, self.inter_channel*self.expansion, kernel_size=1, stride=1, bias=False)
        self.BN_3 = nn.BatchNorm2d(self.inter_channel*self.expansion)
        
        self.relu = nn.ReLU()
        
        self.skip_connect =  nn.Sequential()
        if self.in_channel != self.inter_channel * self.expansion or stride != 1:
            self.skip_connect =  nn.Sequential(nn.Conv2d(self.in_channel, self.inter_channel*self.expansion,
                                                         kernel_size=1, stride=stride))
            
        self.BN_4 = nn.BatchNorm2d(self.inter_channel*self.expansion)
        
    def forward(self, x):
        
        x_in = x
        x = self.conv1x1_1(x)
        x = self.BN_1(x)
        x = self.relu(x)
        
        x = self.conv3x3(x)
        x = self.BN_2(x)
        x = self.relu(x)
        
        x = self.conv1x1_2(x)
        x = self.BN_3(x)
        
        x += self.skip_connect(x_in)
        x = self.BN_4(x)
        x = self.relu(x)
        
        return x
    
class ResNeXt(nn.Module):
    def __init__(self, in_channel=64, out_width=4, C=16, num_classes=10, num_blocks=[3, 4, 6, 3], expansion=2):
        super(ResNeXt, self).__init__()
        self.in_channel = in_channel
        self.C = C
        self.num_classes = num_classes
        self.expansion = expansion
        self.num_block = num_blocks
        self.out_width = out_width
        
        self.layer = nn.Sequential(
            nn.Conv2d(3, self.in_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            self.make_block(1, num_blocks[0]),
            self.make_block(2, num_blocks[1]),
            self.make_block(2, num_blocks[2]),
            
            # Average pooling layer
            nn.AdaptiveAvgPool2d((1, 1)),
            
            # Output FC laye
            nn.Flatten(),
            nn.Linear(self.C*self.out_width, self.num_classes)
        )
        
        
    def make_block(self, stride, num_block):
        blocks = []
        
        for i in range(num_block):
            blocks.append(BottleneckBlock(self.in_channel, self.out_width, self.C, stride, self.expansion))
            self.in_channel =  self.expansion * self.C * self.out_width #2*x*y 64-
        self.out_width = 2*self.out_width #2*x
            
        return nn.Sequential(*blocks)
    
    def forward(self, x):
        x = self.layer(x)
        return x
    