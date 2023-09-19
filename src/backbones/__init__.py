from argparse import Namespace
from datasets import ContinualDatasetInfo
from compressors import CompressorDecompressorInfo
import backbones.mlp as mlp
import backbones.resnet as resnet
import backbones.resnet34 as resnet34
import backbones.resnet18_pretrained as resnet18_pretrained
import backbones.resnet18_cifar as resnet18_cifar
import backbones.resnet32 as resnet32
from torch.nn import Module

ALL_BACKBONES = ['mlp', 'resnet', 'resnet34', 'resnet18_pretrained', 'resnet18_cifar', 'resnet32']

def get_backbone(opts: Namespace, dataset_info: ContinualDatasetInfo, compressor_info: CompressorDecompressorInfo) -> Module:
    """
    returns the backbone object
    
    Args:
        opts: experiment options
        dataset_info: information about the dataset
        compressor_info: information about the compressor
    """

    if opts.backbone == 'mlp':
        return mlp.MLP(opts, dataset_info, compressor_info)
    elif opts.backbone == 'resnet':
        return resnet.Resnet(opts, dataset_info, compressor_info)
    elif opts.backbone == 'resnet34':
        return resnet34.Resnet34(opts, dataset_info, compressor_info)
    elif opts.backbone == 'resnet18_pretrained':
        return resnet18_pretrained.Resnet18_Pretrained(opts, dataset_info, compressor_info)
    elif opts.backbone == 'resnet18_cifar':
        return resnet18_cifar.Resnet18_CIFAR(opts, dataset_info, compressor_info)
    elif opts.backbone == 'resnet32':
        return resnet32.ResNet32(opts, dataset_info, compressor_info)
    else:
        raise ValueError(f'{opts.backbone_name} is not a valid backbone')

def get_backbone_arg_fn(backbone_name: str):
    """
    returns a handle to a funciton that adds backbone-specifig arguments to the argparser
    
    Args:
        backbone_name: name of the backbone, must be in {mlp}
    """
    if backbone_name == 'mlp':
        return mlp.add_backbone_args
    elif backbone_name == 'resnet':
        return resnet.add_backbone_args
    elif backbone_name == 'resnet34':
        return resnet34.add_backbone_args
    elif backbone_name == 'resnet18_pretrained':
        return resnet18_pretrained.add_backbone_args
    elif backbone_name == 'resnet18_cifar':
        return resnet18_cifar.add_backbone_args
    elif backbone_name == 'resnet32':
        return resnet32.add_encoder_args
    else:
        raise ValueError(f'{backbone_name} is not a valid backbone')
    