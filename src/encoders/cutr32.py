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


def add_encoder_args(parser: ArgumentParser) -> None:
    parser.add_argument('--encoding_block', type=int, default=1, required=True, help='after which block to cut the resnet')


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


class Cutr32(nn.Module):
    def __init__(self, opt: Namespace, dataset_info: ContinualDatasetInfo):
        super().__init__()
        block = BasicBlock
        num_classes = dataset_info.num_classes
        assert dataset_info.channels == 3
        assert opt.dataset == 'CIFAR100'
        self.in_planes = 16

        assert opt.encoding_block in [1, 2, 3], f"encoding_block must be 1, 2 or 3, got {opt.encoding_block}"
        self.encoding_block = opt.encoding_block

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, 5, stride=1)
        self.layer2 = self._make_layer(block, 32, 5, stride=2)
        self.layer3 = self._make_layer(block, 64, 5, stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.load_state_dict(torch.load('src/weights/resnet32_cifar100_classes0to49_e374.pt'))
        
        test_tensor = torch.ones([opt.batch_size, 3, dataset_info.size[0], dataset_info.size[1]])
        with torch.no_grad():
            test_output = self(test_tensor)
        
        self.encoder_info = EncoderInfo(
            in_size=dataset_info.size,
            in_channels=3,
            out_size=(test_output.shape[2], test_output.shape[3]),
            out_channels=test_output.shape[1],
            name = f'cutr32_{opt.encoding_block}'
        )
        

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        if self.encoding_block == 1:
            return out
        out = self.layer2(out)
        if self.encoding_block == 2:
            return out
        out = self.layer3(out)
        if self.encoding_block == 3:
            return out    
        assert False, f"encoding_block must be 1, 2 or 3, got {self.encoding_block}"

    def info(self) -> EncoderInfo:
        return self.encoder_info
