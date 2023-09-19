from pathlib import Path
import torch
from torch import nn
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.transforms.functional import normalize
from argparse import ArgumentParser, Namespace
from encoders.encoder import EncoderInfo
from datasets import ContinualDatasetInfo


def add_encoder_args(parser: ArgumentParser) -> None:
    parser.add_argument('--encoding_block', type=int, default=1, required=True, help='the block the resnet is cut at')
    parser.add_argument('--do_rp', action='store_true', help='if provided, perform random projection after CutR-encoding')
    parser.add_argument('--rp_out_dim', type=int, default=-1, help='target dimensions of the random projection. If <=0 rp is not performed.')
    parser.add_argument('--do_normalization', action='store_true', help='if provided, perform normalization on descriptors')
    parser.add_argument('--projection_mat_path', type=str, default='src/encoders/orthogonal_matrices/', help='relative path to the projection matrices')
    parser.add_argument('--normalization_path', type=str, default='src/encoders/descriptor_normalization/', help='relative path to the mean and std for the descritpros. They must be precomputed')


class Cutr34(nn.Module):
    def __init__(self, opt: Namespace, dataset_info: ContinualDatasetInfo):
        """
        Cut Resnet34 with with optional random projection
        """
        super().__init__()

        self.do_rp = opt.do_rp and opt.rp_out_dim >= 0
        self.do_normalization = opt.do_normalization
        self.in_size = dataset_info.size
        self.in_channels = dataset_info.channels

        if self.do_normalization and not self.do_rp:
            raise NotImplementedError()

        cut_layer = _get_cut(opt.encoding_block)
        self.Encoder = nn.Sequential(*list(resnet34(weights=ResNet34_Weights.IMAGENET1K_V1).children()))[:cut_layer].to(opt.device)
        self.Encoder.eval()
        for param in self.Encoder.parameters():
            param.requires_grad = False

        # pass a test tensor through the network to determine the output-shape and size
        if self.in_channels == 1:
            test_tensor = torch.ones((opt.batch_size, 3) + self.in_size).to(opt.device)
        else:
            test_tensor = torch.ones((opt.batch_size, self.in_channels) + self.in_size).to(opt.device)
        cutr_output = self.Encoder(test_tensor)
        cutr_output_shape = cutr_output.shape
        self.cutr_output_numel = cutr_output_shape[1] * cutr_output_shape[2] * cutr_output_shape[3]

        if self.do_rp:
            mat_file = Path(opt.projection_mat_path, f'orthMat_{self.cutr_output_numel}x{opt.rp_out_dim}.pt')
            if mat_file.exists():
                m = torch.load(mat_file)
            else:
                m = torch.nn.init.orthogonal_(torch.empty(self.cutr_output_numel, opt.rp_out_dim))
                torch.save(m, mat_file)
            self.out_chennels = opt.rp_out_dim
            self.out_size = (0, 0)
            self.orth_mat = m.to(opt.device)
            name_tail = f'rp{opt.rp_out_dim}'
        else:
            self.out_chennels = cutr_output_shape[1]
            self.out_size = cutr_output_shape[2:4]
            name_tail = ''

        if self.do_normalization:
            parent = Path(opt.normalization_path)
            self.normalize_mean = torch.load(Path(parent, f'{opt.dataset}_CutR{opt.encoding_block}_rp{opt.rp_out_dim}_mean.pt'))
            self.normalize_std = torch.load(Path(parent, f'{opt.dataset}_CutR{opt.encoding_block}_rp{opt.rp_out_dim}_std.pt'))
        
        self.name = f'cutr34_{opt.encoding_block}' + name_tail

    def forward(self, x: torch.Tensor):
        if self.in_channels == 1:
            x = x.repeat(1, 3, 1, 1)

        assert x.ndim == 4, "Tensor has wrong number of dimensions"
        
        enc = self.Encoder(x)
        if self.do_rp:
            x1 = enc.view(enc.shape[0], 1, self.cutr_output_numel) # right-multiply because we use row-vectors
            x2 = torch.matmul(x1, self.orth_mat)
            x3 = x2.view(x2.size(0), -1)
            if self.do_normalization:
                x4 = x3.view((x3.size(0), x3.size(1), 1, 1))
                x5 = normalize(x4, self.normalize_mean, self.normalize_std)
                x6 = x5.view(x3.size())
                return x6
            else:
                return x3
        else:
            return enc

    def info(self) -> EncoderInfo:
        return EncoderInfo(
            in_size=self.in_size,
            in_channels=self.in_channels,
            out_size=self.out_size,
            out_channels=self.out_chennels,
            name = self.name
        )


def _get_cut(encoding_block: int) -> int:
    """
    determines where to cut in the resnet given the desired block
    
    Args:
        encoding_block: after which block to cut

    Returns:
        index to cut at
    """
    if encoding_block == 1:
        cut_at = 5
    elif encoding_block == 2:
        cut_at = 6
    elif encoding_block == 3:
        cut_at = 7
    elif encoding_block == 4:
        cut_at = 8
    else:
        raise ValueError(f'encodeing_block must be 1,2,3 or 4. Got {encoding_block}')
    return cut_at

