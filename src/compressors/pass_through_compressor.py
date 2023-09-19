from argparse import ArgumentParser, Namespace
from compressors.compressor import CompressorDecompressorInfo
from encoders import EncoderInfo


def add_compressor_args(parser: ArgumentParser):
    pass

class PassThroughCompressor:
    def __init__(self, opt: Namespace, encoder_info: EncoderInfo):
        self.in_size = encoder_info.out_size
        self.in_channels = encoder_info.out_channels
        self.compressed_size = encoder_info.out_size
        self.compressed_channels = encoder_info.out_channels
        self.out_size = encoder_info.out_size
        self.out_channels = encoder_info.out_channels

    def compress(self, x):
        return x

    def decompress(self, x):
        return x

    def info(self):
        return CompressorDecompressorInfo(
            in_size=self.in_size,
            in_channels=self.in_channels,
            compressed_size=self.compressed_size,
            compressed_channels=self.compressed_channels,
            out_size=self.out_size,
            out_channels=self.out_channels
        )
