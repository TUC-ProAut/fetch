from pathlib import Path
import torch
from torch import nn
from torchvision.models import resnet18
from torchvision.transforms.functional import normalize
from argparse import ArgumentParser, Namespace
from encoders.encoder import EncoderInfo
from datasets import ContinualDatasetInfo


def add_encoder_args(parser: ArgumentParser) -> None:
    parser.add_argument('--encoding_block', type=float, default=1, required=True, help='the block the resnet is cut at')


class CutrCIFAR(nn.Module):
    def __init__(self, opt: Namespace, dataset_info: ContinualDatasetInfo):
        """
        Cut Resnet, pretrained on the first two CIFAR10 Classes
        """
        super().__init__()

        self.in_size = dataset_info.size
        self.in_channels = dataset_info.channels

        assert Path('src/weights/resnet18_cifar10_classes01_e110.pt').is_file()
        resnet = resnet18()
        resnet.fc = torch.nn.Linear(512, 10)
        resnet.load_state_dict(torch.load('src/weights/resnet18_cifar10_classes01_e110.pt'))
        
        cut_layer = _get_cut(opt.encoding_block)
        if opt.encoding_block in [1, 2, 3, 4]:
            self.Encoder = nn.Sequential(*list(resnet.children())[:cut_layer]).to(opt.device)
        elif opt.encoding_block in [2.5, 3.5]:
            # cut the last resnet-block in half
            self.Encoder = torch.nn.Sequential(
                *(
                    list(resnet.children())[:cut_layer]
                    + [list(list(resnet.children())[cut_layer].children())[0]]
                )
            ).to(opt.device)
        else:
            raise ValueError(f'encodeing_block must be 1,2,3, 3.5 or 4. Got {opt.encoding_block}')
            
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

        
        self.out_chennels = cutr_output_shape[1]
        self.out_size = cutr_output_shape[2:4]
        
        self.name = f'cutr_cifar{opt.encoding_block}'

    def forward(self, x: torch.Tensor):
        if self.in_channels == 1:
            x = x.repeat(1, 3, 1, 1)

        assert x.ndim == 4, "Tensor has wrong number of dimensions"
        
        enc = self.Encoder(x)
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
    elif encoding_block == 2 or encoding_block == 2.5:
        cut_at = 6
    elif encoding_block == 3 or encoding_block == 3.5:
        cut_at = 7
    elif encoding_block == 4:
        cut_at = 8
    return cut_at

