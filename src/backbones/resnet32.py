# see https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py
# 
# Copyright (c) 2018, Yerlan Idelbayev
# 
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from argparse import Namespace, ArgumentParser
from encoders.encoder import EncoderInfo
from datasets import ContinualDatasetInfo
from compressors.compressor import CompressorDecompressorInfo


def add_encoder_args(parser: ArgumentParser) -> None:
    parser.add_argument('--backbone_block', type=int, default=1, required=True, help='after which block to cut the resnet')
    parser.add_argument('--dont_pretrain_backbone', action="store_true", help="Dont load pretraining weights into the state dict of the encoder")


def _weights_init(m):
    """He initialization"""
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet32(nn.Module):
    """ResNet32, pretrained on the the first 50 CIFAR100 classes"""
    def __init__(self, opt: Namespace, dataset_info: ContinualDatasetInfo, compressor_info: CompressorDecompressorInfo):
        super().__init__()
        block = BasicBlock
        num_classes = dataset_info.num_classes
        self.in_planes = 16
        assert opt.dataset == 'CIFAR100'
        assert opt.backbone_block in [0, 1, 2, 3], f"backbone_block must be 1, 2 or 3, got {opt.backbone_block}"
        self.backbone_block = opt.backbone_block

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, 5, stride=1)
        self.layer2 = self._make_layer(block, 32, 5, stride=2)
        self.layer3 = self._make_layer(block, 64, 5, stride=2)
        self.linear = nn.Linear(64, num_classes)

        if opt.dont_pretrain_backbone:
            self.apply(_weights_init)
        else:
            self.load_state_dict(torch.load('src/weights/resnet32_cifar100_classes0to49_e374.pt'))
            print('Loaded Weights')

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        assert self.backbone_block < 4, f"backbone_block must be 0, 1, 2 or 3, got {self.backbone_block}"
        if self.backbone_block == 0:
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.layer1(x)
        if self.backbone_block <= 1:
            x = self.layer2(x)
        if self.backbone_block <= 2:
            x = self.layer3(x)

        x = F.avg_pool2d(x, x.size()[3])
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x
