# see https://medium.com/pytorch/implementing-an-autoencoder-in-pytorch-19baa22647d1
# and https://analyticsindiamag.com/how-to-implement-convolutional-autoencoder-in-pytorch-with-cuda/

from typing import Callable, Any, Tuple
from pathlib import Path
from argparse import ArgumentParser
import torch.nn as nn
import torch.nn.functional as F
import torch
from encoders import EncoderInfo
from compressors import CompressorDecompressorInfo


def add_compressor_args(parser: ArgumentParser) -> None:
    parser.add_argument('--latent_channels', type=int, required=True, help='Number of channels in the latent vector. Note that the vector will have spacial dimensions as well.a')
    parser.add_argument('--n_blocks', type=int, default=2, help='How many times conv-pool is applied. Must be greater than two')
    parser.add_argument('--pretraining_params', type=str, default='TinyImagenet', help='on which dataset to pretrain the conv-autoencoder. Possible are "TinyImagenet" or "CIFAR10_01"')


class _Encoding_Module(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = F.relu(x)
        x = self.pool(x)
        return x


class _Decoding_Module(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, relu: bool):
        super().__init__()
        self.t_conf = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2, 
            stride=2)
        self.relu = relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.t_conf(x)
        if self.relu:
            x = F.relu(x)
        return x


class ConvAutoencoder(nn.Module):
    def __init__(self, opt, encoder_info: EncoderInfo, train_ae=False):
        """Consits of --n_blocks modules containing conv and pool(2, 2)-blocks."""
        super().__init__()

        assert encoder_info.out_size != (0, 0)
        if opt.n_blocks < 2:
            raise ValueError(f"--n_blocks must be greater than 2, got {opt.n_blocks}")

        self.data_size = encoder_info.out_size
        self.data_channels = encoder_info.out_channels

        # Make the compressor and decompressor
        n_mid_channels = max(int(abs(opt.latent_channels + encoder_info.out_channels) / 2), 16)

        enc_modules = []
        for i in range(opt.n_blocks):
            if i == 0:
                in_channels = self.data_channels
                out_channels = n_mid_channels
            elif i == opt.n_blocks - 1:
                in_channels = n_mid_channels
                out_channels = opt.latent_channels
            else:
                in_channels = n_mid_channels
                out_channels = n_mid_channels
            enc_modules.append(_Encoding_Module(in_channels=in_channels, out_channels=out_channels).to(opt.device))
        
        dec_modules = []
        for i in range(opt.n_blocks):
            if i == 0:
                in_channels = opt.latent_channels
                out_channels = n_mid_channels
                relu = True
            elif i == opt.n_blocks -1:
                in_channels = n_mid_channels
                out_channels = self.data_channels
                relu = False
            else:
                in_channels = n_mid_channels
                out_channels = n_mid_channels
                relu = True
            dec_modules.append(_Decoding_Module(in_channels=in_channels, out_channels=out_channels, relu=relu).to(opt.device))            
        self.compressor = nn.Sequential(*enc_modules)  # type: ignore
        self.decompressor = nn.Sequential(*dec_modules)  # type: ignore

        # load parameters
        if not train_ae:
            if opt.pretraining_params == 'TinyImagenet':
                param_filename = f'autoencoder_TinyImagenet_{encoder_info.name}_lc_{opt.latent_channels}_nb{opt.n_blocks}.pt'
            elif opt.pretraining_params == 'CIFAR10_01':
                assert opt.n_blocks == 2, "Conv-AE params only valid for autoencoders with 2 blocks"
                param_filename = f'autoencoder_cifar10_classes_01_{encoder_info.name}_lc_{opt.latent_channels}_nb{opt.n_blocks}.pt'
            else:
                raise ValueError(f"no such parameters for the convolutional autoencoder: {opt.pretraining_params}")
            param_file = Path('src', 'compressors', 'compressor_params', param_filename).resolve()
            if param_file.exists():
                self.load_state_dict(torch.load(param_file, map_location='cpu'))
            else:
                raise FileNotFoundError(f'Param File {param_file} was not found for this configuration. The Autoencoder must be trained before the experiment!')
        
        # test out the compressor
        self.to(opt.device)
        self.compressed_channels, self.compressed_shape = self._get_compressed_shape(opt.device)

    def _get_compressed_shape(self, device: str) -> Tuple[int, Tuple[int, int]]:
        tt = torch.ones([1, self.data_channels, self.data_size[0], self.data_size[1]]).to(device)
        c = self.compress(tt)
        d = self.decompress(c)
        assert tt.shape == d.shape, "Something went wrong, Input and Output have different shapes"
        channels = c.size(1)
        shape = c.size(2), c.size(3)
        return channels, shape

    def compress(self, x):
        return self.compressor(x)

    def decompress(self, x):
        return self.decompressor(x)

    def forward(self, x):
        x = self.compress(x)
        x = self.decompress(x)
        return x

    def info(self) -> CompressorDecompressorInfo:
        return CompressorDecompressorInfo(
            in_size=self.data_size,
            in_channels=self.data_channels,
            compressed_size=self.compressed_shape,
            compressed_channels=self.compressed_channels,
            out_size=self.data_size,
            out_channels=self.data_channels
        )
