# see https://medium.com/pytorch/implementing-an-autoencoder-in-pytorch-19baa22647d1
# and https://analyticsindiamag.com/how-to-implement-convolutional-autoencoder-in-pytorch-with-cuda/

from argparse import ArgumentParser
from pathlib import Path
import torch.nn as nn
import torch
from encoders import EncoderInfo
from compressors import CompressorDecompressorInfo


def add_compressor_args(parser: ArgumentParser) -> None:
    parser.add_argument('--n_layers', type=int, default=6, help='How many times conv-pool is applied. Must be greater than two')
    parser.add_argument('--bottleneck_neurons', type=int, required=True, help='How many neurons there are in the bottleneck')


class _Fc_Block(nn.Module):
    def __init__(self, in_features: int, out_features: int, do_relu: bool=True):
        super().__init__()
        if do_relu:
            self.model = nn.Sequential(
                nn.Linear(in_features=in_features, out_features=out_features),
                nn.ReLU(True)
            )
        else:
            self.model = nn.Linear(in_features=in_features, out_features=out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class FcAutoencoder(nn.Module):
    def __init__(self, opt, encoder_info: EncoderInfo, train_ae=False):
        super().__init__()

        assert encoder_info.out_size[0] != 0 and encoder_info.out_size[1] != 0

        self.data_size = encoder_info.out_size
        self.data_channels = encoder_info.out_channels
        self.bottleneck_neurons = opt.bottleneck_neurons

        # compute the layout of the architecture
        data_numel = self.data_channels * self.data_size[0] * self.data_size[1]
        assert data_numel > opt.bottleneck_neurons
        down_factor = (opt.bottleneck_neurons / data_numel) ** (-1/opt.n_layers)
        n_features_list = [round(data_numel / (down_factor**i)) for i in range(opt.n_layers)] + [self.bottleneck_neurons]

        # build the architecture
        compressor_blocks = []
        for i_channel in range(opt.n_layers):
            compressor_blocks.append(_Fc_Block(
                in_features=n_features_list[i_channel],
                out_features=n_features_list[i_channel+1]
            ).to(opt.device))
        self.compressor = nn.Sequential(*compressor_blocks)
        decompressor_blocks = []
        for i_channel in range(opt.n_layers, 0, -1):
            do_relu = i_channel != 1
            decompressor_blocks.append(_Fc_Block(
                in_features=n_features_list[i_channel],
                out_features=n_features_list[i_channel-1],
                do_relu=do_relu
            ).to(opt.device))
        self.decompressor = nn.Sequential(*decompressor_blocks)
        
        # load the parameters
        self.train_ae = train_ae
        if not train_ae:
            param_filename = f'fcae_TinyImagenet_{self.data_size[0]}x{self.data_size[1]}_{encoder_info.name}_ls{opt.bottleneck_neurons}.pt'
            param_file = Path('src', 'compressors', 'compressor_params', param_filename).resolve()
            if param_file.exists():
                self.load_state_dict(torch.load(param_file))
            else:
                raise FileNotFoundError(f'Param File {param_file} was not found for this configuration. The Autoencoder must be trained before the experiment!')

    def compress(self, x: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(self.train_ae):
            bs = x.size(0)
            flattened = x.view(bs, -1)
            comp = self.compressor(flattened)
        return comp

    def decompress(self, x: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(self.train_ae):
            bs = x.size(0)
            flattened = self.decompressor(x)
            x_hat = flattened.view(bs, self.data_channels, self.data_size[0], self.data_size[1])
        return x_hat

    def forward(self, x):
        assert self.train_ae, 'Be sure to deactivate train_ae when training!'
        z = self.compress(x)
        x_hat = self.decompress(z)
        return x_hat

    def info(self) -> CompressorDecompressorInfo:
        return CompressorDecompressorInfo(
            in_size=(self.data_size[0], self.data_size[1]),
            in_channels=self.data_channels,
            compressed_size=(0, 0),
            compressed_channels=self.bottleneck_neurons,
            out_size=(self.data_size[0], self.data_size[1]),
            out_channels=self.data_channels,
        )