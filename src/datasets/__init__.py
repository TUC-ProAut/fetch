from argparse import Namespace
from datasets.continual_dataset import *
import datasets.cifar10
import datasets.cifar100
import datasets.mnist
import datasets.tiny_imagenet
import datasets.core50

ALL_DATASETS = ['MNIST', 'CIFAR10', 'CIFAR100', 'TinyImagenet', 'CORe50', 'CORe50@1fps']


def get_dataset(opt: Namespace) -> ContinualDataset:
    """returns a dataset specified by name
    
    Args:
        opt: Experiment Configuration
    """
    if opt.dataset == 'MNIST':
        return datasets.mnist.ContinualMnist(opt) #type: ignore
    elif opt.dataset == 'CIFAR10':
        return datasets.cifar10.ContinualCifar10(opt) #type: ignore
    elif opt.dataset == 'CIFAR100':
        return datasets.cifar100.ContinualCifar100(opt) #type: ignore
    elif opt.dataset == 'TinyImagenet':
        return datasets.tiny_imagenet.ContinualTinyImagenet(opt) #type: ignore
    elif opt.dataset == 'CORe50':
        return datasets.core50.Core50(opt, reduce_fps=False) #type: ignore
    elif opt.dataset == 'CORe50@1fps':
        return datasets.core50.Core50(opt, reduce_fps=True) #type: ignore
    else:
        raise ValueError(f'{opt.dataset} is not a valid dataset')