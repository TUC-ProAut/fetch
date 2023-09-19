# see https://github.com/francesco-p/smaller-is-better/blob/master/src/models/vae.py

from pathlib import Path
from typing import Tuple
from argparse import ArgumentParser
from torch.nn import functional as F
import torch.nn as nn
import torch
from encoders import EncoderInfo
from compressors import CompressorDecompressorInfo


def add_compressor_args(parser: ArgumentParser) -> None:
    parser.add_argument('--latent_size', type=int, required=True, help='Number of dimensions in the vae-latent representation')
    parser.add_argument('--vae_width', type=int, required=False, default=400, help='Width of the hidden layer in the vae')


def vae_loss_function(recon_x: torch.Tensor, x, mu, logvar):
    """Reconstruction (binary-cross-entrpty) + KLD summed over all elements and batch

    Params:
        recon_x: Reconstructed data
        x: original input-data
        mu: mu-output of the vae-encoder
        logvar: log(sigma)-output of the vae-encoder

    Returns:
        Loss (one-dimensional)
    """
    assert recon_x.ndim == 4
    bs = recon_x.size(0)
    BCE = F.binary_cross_entropy(recon_x.view(bs, -1), x.view(bs, -1), reduction='sum')
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # here logvar := log(sigma)
    KLD = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0)
    return BCE + KLD


class VAE1C(nn.Module):
    def __init__(self, opt, encoder_info: EncoderInfo, train_vae=False):
        super().__init__()

        assert encoder_info.in_size != (0, 0)
        self.data_size = encoder_info.in_size
        self.data_channels = encoder_info.in_channels
        self.data_numel = self.data_size[0] * self.data_size[1] * self.data_channels

        if not train_vae:
            param_filename = f'vae_{opt.dataset}_{encoder_info.name}_width{opt.vae_width}_emb{opt.latent_size}.pt'
            param_file = Path('compressors', 'compressor_params', param_filename).resolve()
            if param_file.exists():
                self.load_state_dict(torch.load(param_file))
            else:
                raise FileNotFoundError(f'Param File {param_file} was not found for this configuration. The VAE must be trained vefore the experiment!')

        self.fc1 = nn.Linear(self.data_numel, opt.vae_width)
        self.fc21 = nn.Linear(opt.vae_width, opt.latent_size) # <----- mu
        self.fc22 = nn.Linear(opt.vae_width, opt.latent_size) # <----- log(sigma)

        self.fc3 = nn.Linear(opt.latent_size, opt.vae_width) 
        self.fc4 = nn.Linear(opt.vae_width, self.data_numel)

    def get_gauss_params(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """returns tuple mu, log(sigma) for the input x
        """
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        """samples form the gaussian given by mu, log(sigma)
        """
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def compress(self, x: torch.Tensor) -> torch.Tensor:
        """represent a sample as a latent vector (sampling from gaussian is performed)
        """
        mu, logvar = self.get_gauss_params(x)
        z = self.reparameterize(mu, logvar)
        return z

    def decompress(self, z):
        h3 = F.relu(self.fc3(z))
        h3s = torch.sigmoid(self.fc4(h3))
        return h3s.view(-1, self.data_channels, self.data_size[0], self.data_size[1])

    def forward(self, x):
        mu, logvar = self.get_gauss_params(x.view(-1, self.data_numel))
        z = self.reparameterize(mu, logvar)
        return self.decompress(z), mu, logvar

    def info(self):
        return CompressorDecompressorInfo(
            in_size=self.data_size,
            in_channels=self.data_channels,
            compressed_size=(0, 0),
            compressed_channels=self.data_channels,
            out_size=self.data_size,
            out_channels=self.data_channels
        )
