from argparse import ArgumentParser, Namespace
from ast import comprehension
import torch.nn as nn
from backbones.layers import FCBlock, FinalBlock
from datasets import ContinualDatasetInfo
from compressors import CompressorDecompressorInfo

def add_backbone_args(parser: ArgumentParser):
    parser.add_argument('--width', type=int, default=400, help='Width of a model')

class MLP(nn.Module):
    def __init__(self, opts: Namespace, dataset_info: ContinualDatasetInfo, compressor_info: CompressorDecompressorInfo):
        super(MLP, self).__init__()

        input = FCBlock(opt=opts, in_channels=compressor_info.out_channels, out_channels=opts.width)
        hidden1 = FCBlock(opt=opts, in_channels=opts.width, out_channels=opts.width)
        final = FinalBlock(num_classes=dataset_info.num_classes, opt=opts, in_channels=opts.width)

        self.net = nn.Sequential(input, hidden1, final)

    def forward(self, x):
        return self.net(x)

