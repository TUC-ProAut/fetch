from argparse import ArgumentParser, Namespace
import torch
import torch.nn as nn

from datasets import ContinualDatasetInfo
from compressors import CompressorDecompressorInfo
from backbones.layers import ConvBlock, FinalBlock



def add_backbone_args(parser: ArgumentParser):
    parser.add_argument('--backbone_block', type=float, default=0, required=False, help='after which block to cut the resnet. Use 0 to use the whole resnet')


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, opt, inChannels, outChannels, stride=1, downsample=None):
        """Building Block of resnet consisting of two conv-layers and a shortcut"""
        super().__init__()
        self.downsample = downsample
        expansion = 1
        self.conv1 = ConvBlock(opt=opt, in_channels=inChannels, out_channels=outChannels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = ConvBlock(opt=opt, in_channels=outChannels, out_channels=outChannels*expansion, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        _out = self.conv1(x)
        _out = self.conv2(_out)
        if self.downsample is not None:
            shortcut = self.downsample(x)
        else:
            shortcut = x
        _out = _out + shortcut
        return _out

class ResidualBlock(nn.Module):
    def __init__(self, opt, block, inChannels, outChannels, depth, stride=1):
        super().__init__()
        if stride != 1 or inChannels != outChannels * block.expansion:
            downsample = ConvBlock(opt=opt, in_channels=inChannels, out_channels=outChannels* block.expansion, kernel_size=1, stride=stride, padding=0, bias=False)
        else:
            downsample = None
        self.blocks = nn.Sequential()
        self.blocks.add_module('block0', block(opt, inChannels, outChannels, stride, downsample))
        inChannels = outChannels * block.expansion
        for i in range(1, depth):
            self.blocks.add_module('block{}'.format(i), block(opt, inChannels, outChannels))

    def forward(self, x):
        return self.blocks(x)

class PartialResidualBlock(nn.Module):
    def __init__(self, opt, block, inChannels, outChannels, depth, stride=1):
        assert depth==2
        super().__init__()
        self.blocks = nn.Sequential()
        inChannels = outChannels * block.expansion
        for i in range(1, depth):
            self.blocks.add_module('block{}'.format(i), block(opt, inChannels, outChannels))

    def forward(self, x):
        return self.blocks(x)



class Resnet(nn.Module):
    def __init__(self, opt: Namespace, dataset_info: ContinualDatasetInfo, compressor_info: CompressorDecompressorInfo):
        """resnet 18. It is possible to cut off layers form the front."""
        super().__init__()
        self.cut_layer = opt.backbone_block
        num_blocks = [2, 2, 2, 2]
        block = BasicBlock
        in_planes, out_planes = 64, 512 #20, 160

        self.num_classes = dataset_info.num_classes
        initial = ConvBlock(opt=opt, in_channels=compressor_info.out_channels, out_channels=in_planes, kernel_size=3, stride=1, padding=1)
        group1 = ResidualBlock(opt=opt, block=block, inChannels=64, outChannels=64, depth=num_blocks[0], stride=1) #For ResNet-S, convert this to 20,20
        group2 = ResidualBlock(opt=opt, block=block, inChannels=64*block.expansion, outChannels=128, depth=num_blocks[1], stride=2) #For ResNet-S, convert this to 20,40
        group3 = ResidualBlock(opt=opt, block=block, inChannels=128*block.expansion, outChannels=256, depth=num_blocks[2], stride=2) #For ResNet-S, convert this to 40,80
        group3_partial = PartialResidualBlock(opt=opt, block=block, inChannels=128*block.expansion, outChannels=256, depth=num_blocks[2], stride=2) #For ResNet-S, convert this to 40,80
        group4 = ResidualBlock(opt=opt, block=block, inChannels=256*block.expansion, outChannels=512, depth=num_blocks[3], stride=2) #For ResNet-S, convert this to 80,160
        group4_partial = PartialResidualBlock(opt=opt, block=block, inChannels=256*block.expansion, outChannels=512, depth=num_blocks[3], stride=2) #For ResNet-S, convert this to 80,160
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.final = FinalBlock(num_classes=dataset_info.num_classes, opt=opt, in_channels=out_planes*block.expansion)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if self.cut_layer == 0:
            self.blocks = nn.Sequential(
                initial,
                group1,
                group2,
                group3,
                group4
            )
        elif self.cut_layer == 1:
            self.blocks = nn.Sequential(
                group2,
                group3,
                group4
            )
        elif self.cut_layer == 2:
            self.blocks = nn.Sequential(
                group3,
                group4
            )
        elif self.cut_layer == 2.5:
            self.blocks = nn.Sequential(
                group3_partial,
                group4
            )
        elif self.cut_layer == 3:
            self.blocks = nn.Sequential(
                group4
            )
        elif self.cut_layer == 3.5:
            self.blocks = nn.Sequential(
                group4_partial
            )
        else:
            raise ValueError(f'--backbone_block must be 0, 1, 2 or 3. Got {self.cut_layer}')


    def forward(self, x):
        out = self.blocks(x)
        
        out = self.pool(out)
        out = out.view(x.size(0), -1)
        out = self.final(out)
        return out