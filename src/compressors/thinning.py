from argparse import ArgumentParser, Namespace
from math import ceil
import torch
from compressors.compressor import CompressorDecompressorInfo
from encoders import EncoderInfo


def add_compressor_args(parser: ArgumentParser):
    parser.add_argument('--compression_factor', type=float, required=True, help='How much smaller the data is after compression. Use values [0, 1[')


class Thinning:
    def __init__(self, opt: Namespace, encoder_info: EncoderInfo):
        """only keep the (1-x)% largest values in every tensor. x is the compression-factor.
        """
        self.in_size = encoder_info.out_size
        self.in_channels = encoder_info.out_channels
        self.compressed_size = encoder_info.out_size
        self.compressed_channels = encoder_info.out_channels
        self.out_size = encoder_info.out_size
        self.out_channels = encoder_info.out_channels
        
        assert encoder_info.out_size[0] != 0 and encoder_info.out_size[1] != 0
        assert encoder_info.out_channels != 0
        assert opt.compression_factor >= 0 and opt.compression_factor < 1

        len_sample = encoder_info.out_size[0] * encoder_info.out_size[1] * encoder_info.out_channels
        n_del = ceil(len_sample * opt.compression_factor)
        if n_del == len_sample:
            n_del = len_sample - 1

        self.n_del = n_del
        self.proportion_del = n_del / len_sample

    def compress(self, x):
        assert x.ndim >= 4
        orig_shape = x.shape

        x2 = x.view(orig_shape[0], -1)
        x3, idx_sort = x2.sort(1)
        idx_unsort = torch.argsort(idx_sort)
        x3[:, :self.n_del] = 0
        x4 = x3.gather(1, idx_unsort)
        x5 = x4.view(orig_shape)
        return x5

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
