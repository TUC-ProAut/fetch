from argparse import Namespace
from datasets import ContinualDatasetInfo
import encoders.cutr as cutr
import encoders.cutr34 as cutr34
import encoders.cutr_cifar as cutr_cifar
import encoders.cutr32 as cutr32
import encoders.pass_through_encoder as pass_through_encoder
from encoders.encoder import Encoder, EncoderInfo


ALL_ENCODERS = ['none', 'cutr', 'cutr34', 'cutr_cifar', 'cutr32']


def get_encoder(opts: Namespace, dataset_info: ContinualDatasetInfo) -> Encoder:
    """
    returns the encoder object

    Args
        opts: experiment options
        dataset: dataset object (only to read some information)
    """
    if opts.encoder == 'cutr':
        return cutr.Cutr(opts, dataset_info) #type: ignore
    elif opts.encoder == 'cutr34':
        return cutr34.Cutr34(opts, dataset_info) #type: ignore
    elif opts.encoder == 'cutr_cifar':
        return cutr_cifar.CutrCIFAR(opts, dataset_info) #type: ignore
    elif opts.encoder == 'cutr32':
        return cutr32.Cutr32(opts, dataset_info) #type: ignore
    elif opts.encoder == 'none':
        return pass_through_encoder.PassThroughEncoder(opts, dataset_info) #type: ignore
    else:
        raise ValueError(f'{opts.encoder} is not a valid encoder')


def get_encoder_arg_fn(encoder_name: str):
    """
    returns a handle to a funciton that adds encoder-specifig arguments to the argparser
    
    Args:
        encoder_name: name of the encoder, must be in {cutr}
    """
    if encoder_name == 'cutr':
        return cutr.add_encoder_args
    elif encoder_name == 'cutr34':
        return cutr34.add_encoder_args
    elif encoder_name == 'cutr_cifar':
        return cutr_cifar.add_encoder_args
    elif encoder_name == 'cutr32':
        return cutr32.add_encoder_args
    elif encoder_name == 'none':
        return pass_through_encoder.add_encoder_args
    else:
        raise ValueError(f'{encoder_name} is not a valid encoder')
    