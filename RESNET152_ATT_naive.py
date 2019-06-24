# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU6(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * 4)
        self.relu = nn.ReLU6(inplace=True)
        self.downsample = downsample
        #ATT
        self.att = nn.Conv1d(planes,1,kernel_size=1,bias=True)
        self.sigmoid = nn.Sigmoid()
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        #ATT
        out_att = self.att(out)
        out_att = self.sigmoid(out_att)
        out = out*out_att

        out += residual
        out = self.relu(out)

        return out
    
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=15):
        self.inplanes=64
        groupNum=1
        super(ResNet,self).__init__()
        self.conv1=nn.Conv1d(3,64,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn1=nn.BatchNorm1d(64)
        #ATT
        self.att=nn.Conv1d(64,1,kernel_size=1,bias=True)
        self.relu = nn.ReLU6(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64*groupNum, layers[0])
        self.layer2 = self._make_layer(block, 128*groupNum, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256*groupNum, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512*groupNum, layers[3], stride=2)
        self.avgpool = nn.AvgPool1d(5)
        self.fc = nn.Linear(512*groupNum * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = list()
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        #ATT
        x_att = self.att(x)
        x = x*x_att

        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        embed=x
        x = self.fc(x)

        return F.log_softmax(x),embed,x_att#input=x,dim=1



def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model

def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3,4,6,3], **kwargs)
    return model

def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3,4,23,3], **kwargs)
    return model
    
def resnet152(**kwargs):
    """Constructs a ResNet-152 model.
    """
    model = ResNet(Bottleneck, [3,8,36,3], **kwargs)
    return model
