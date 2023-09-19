from argparse import Namespace
from encoders import EncoderInfo
from compressors.compressor import CompressorDecompressorInfo, CompressorDecompressor
import compressors.pass_through_compressor as pass_through_compressor
import compressors.thinning as thinning
import compressors.quantization as quantization
import compressors.vae as vae
import compressors.conv_autoencoder as conv_autoencoder
import compressors.fc_autoencoder as fc_autoencoder

ALL_COMPRESSORS = ['none', 'thinning', 'quantization', 'vae', 'convae', 'fcae']

def get_compressor(opts: Namespace, encoder_info: EncoderInfo) -> CompressorDecompressor:
    """
    returns the object that handles compression and decompression

    Args
        opts: experiment options
        encoder_info: information about the encoder
    """
    if opts.compressor == 'none':
        return pass_through_compressor.PassThroughCompressor(opts, encoder_info) #type: ignore
    elif opts.compressor == 'thinning':
        return thinning.Thinning(opts, encoder_info)    #type: ignore
    elif opts.compressor == 'quantization':
        return quantization.Quantization(opts, encoder_info) #type: ignore
    elif opts.compressor == 'convae':
        return conv_autoencoder.ConvAutoencoder(opts, encoder_info)    #type: ignore
    elif opts.compressor == 'vae':
        return vae.VAE1C(opts, encoder_info)    #type: ignore
    elif opts.compressor == 'fcae':
        return fc_autoencoder.FcAutoencoder(opts, encoder_info) # type: ignore
    else:
        raise ValueError(f'{opts.compressor} is not a valid compressor')


def get_compressor_arg_fn(compressor_name: str):
    """
    returns a handle to a funciton that adds encoder-specifig arguments to the argparser
    
    Args:
        compressor_name: name of the compressor
    """
    if compressor_name == 'none':
        return pass_through_compressor.add_compressor_args  # type: ignore
    elif compressor_name == 'thinning':
        return thinning.add_compressor_args # type: ignore
    elif compressor_name == 'quantization':
        return quantization.add_compressor_args
    elif compressor_name == 'convae':
        return conv_autoencoder.add_compressor_args
    elif compressor_name == 'vae':
        return vae.add_compressor_args
    elif compressor_name == 'fcae':
        return fc_autoencoder.add_compressor_args
    else:
        raise ValueError(f'{compressor_name} is not a valid compressor')
    