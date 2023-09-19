from torch import nn
from argparse import ArgumentParser, Namespace
from encoders.encoder import EncoderInfo
from datasets import ContinualDatasetInfo

def add_encoder_args(parser: ArgumentParser) -> None:
    pass


class PassThroughEncoder(nn.Module):
    def __init__(self, opts: Namespace, dataset_info: ContinualDatasetInfo):
        """
        Do not perform any encoding
        """
        super().__init__()
        self.in_size = dataset_info.size
        self.in_channels = dataset_info.channels
        self.out_size = self.in_size
        self.out_channels = self.in_channels

    def forward(self, x):
        return x

    def info(self) -> EncoderInfo:
        return EncoderInfo(
            in_size=self.in_size,
            in_channels=self.in_channels,
            out_size=self.out_size,
            out_channels=self.out_channels,
            name='passThrough'
        )
