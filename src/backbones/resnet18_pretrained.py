from argparse import ArgumentParser, Namespace
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

from datasets import ContinualDatasetInfo
from compressors import CompressorDecompressorInfo
from backbones.layers import FinalBlock



def add_backbone_args(parser: ArgumentParser):
    parser.add_argument('--backbone_block', type=float, default=0, required=False, help='after which block to cut the resnet. Use 0 to use the whole resnet')


class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x


class Resnet18_Pretrained(nn.Module):
    def __init__(self, opt: Namespace, dataset_info: ContinualDatasetInfo, compressor_info: CompressorDecompressorInfo):
        super().__init__()
        full_resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        full_resnet.fc = torch.nn.Linear(512, 10)
        if opt.backbone_block == 0:
            body = torch.nn.Sequential(*list(full_resnet.children())[:-1])
        else:
            cut_layer = _get_cut(opt.backbone_block)
            if opt.backbone_block in [1, 2, 3, 4]:
                body = torch.nn.Sequential(*list(full_resnet.children())[cut_layer:-1])
            elif opt.backbone_block in [2.5, 3.5]:
                body = torch.nn.Sequential(
                    *(
                        [list(list(full_resnet.children())[cut_layer].children())[1]]
                        + list(full_resnet.children())[cut_layer+1:-1]
                    )
                )
                
        # https://discuss.pytorch.org/t/break-resnet-into-two-parts/39315/2
        fc = torch.nn.Sequential(
            Flatten(),
            FinalBlock(num_classes=dataset_info.num_classes, opt=opt, in_channels=512)
        )
        self.Backbone = torch.nn.Sequential(
            body,
            fc
        )

    def forward(self, x):
        out = self.Backbone(x)
        return out




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
    else:
        raise ValueError(f'encodeing_block must be 1,2,3 or 4. Got {encoding_block}')
    return cut_at