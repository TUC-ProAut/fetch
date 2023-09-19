import torch, time
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
from opts import add_general_args, parse_general_args
from utils import AverageMeter, get_accuracy, get_logger, seed_everything, cutmix_data #type: ignore

from datasets import get_dataset, ContinualDataset
from encoders import get_encoder, get_encoder_arg_fn, Encoder
from compressors import get_compressor, get_compressor_arg_fn, CompressorDecompressor
from backbones import get_backbone, get_backbone_arg_fn
from samplers import get_sampler, get_sampler_arg_fn, Sampler

def test(opt: argparse.Namespace,
         encoder: Encoder,
         loader: DataLoader,
         model: nn.Module,
         criterion,
         class_mask: np.ndarray,
         logger,
         epoch: int):

        model.eval()
        losses = AverageMeter()
        batch_time = AverageMeter()
        accuracy = AverageMeter()
        task_accuracy = AverageMeter()

        with torch.no_grad():
            start = time.time()
            for inputs, labels in loader:
                inputs = inputs.to(opt.device)
                labels = labels.to(opt.device)

                inputs = encoder(inputs)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                losses.update(loss.data, inputs.size(0))

                # Measure accuracy and task accuracy 
                prob = torch.nn.functional.softmax(outputs, dim=1)  # type: ignore
                acc, task_acc = get_accuracy(prob, labels, class_mask)
                accuracy.update(acc, labels.size(0))
                task_accuracy.update(task_acc, labels.size(0))
                batch_time.update(time.time() - start)
                start = time.time()

        logger.info('==> Test: [{0}]\tTime:{batch_time.sum:.4f}\tLoss:{losses.avg:.4f}\tAcc:{acc.avg:.4f}\tTask Acc:{task_acc.avg:.4f}\t'
            .format(epoch, batch_time=batch_time, losses=losses, acc=accuracy, task_acc=task_accuracy))
        return accuracy.avg


def train(opt: argparse.Namespace,
          loader: DataLoader,
          model: nn.Module,
          compressor: CompressorDecompressor,
          criterion,
          optimizer,
          epoch: int,
          logger):

        model.train()
        losses = AverageMeter()
        data_time = AverageMeter()
        batch_time = AverageMeter()


        start = time.time()

        for inputs, labels in loader:
            inputs = inputs.to(opt.device)
            labels = labels.to(opt.device)

            do_cutmix = opt.regularization == 'cutmix' and np.random.rand(1) < opt.cutmix_prob
            if do_cutmix:
                inputs, labels_a, labels_b, lam = cutmix_data(x=inputs, y=labels, alpha=opt.cutmix_alpha)

            data_time.update(time.time() - start)

            decompressed_inputs = compressor.decompress(inputs)
            
            # Forward, backward passes then step
            outputs = model(decompressed_inputs)
            if do_cutmix:
                loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
            else:
                loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip) #type: ignore
            optimizer.step()

            # Log losses
            losses.update(loss.data.item(), labels.size(0))
            batch_time.update(time.time() - start)
            start = time.time()

        logger.info('==> Train:[{0}]\tTime:{batch_time.sum:.4f}\tData:{data_time.sum:.4f}\tLoss:{loss.avg:.4f}\t'
            .format(epoch, batch_time=batch_time, data_time=data_time, loss=losses))
        return model, optimizer


def experiment(opt: argparse.Namespace,
               dataset: ContinualDataset,
               encoder: Encoder,
               compressor: CompressorDecompressor, 
               memory: Sampler,
               backbone: nn.Module, 
               logger):

    encoder = encoder.to(opt.device)
    backbone = backbone.to(opt.device)
    criterion = nn.CrossEntropyLoss().to(opt.device)
    optimizer = optim.SGD(backbone.parameters(), lr=opt.maxlr, momentum=0.9, weight_decay=opt.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2, eta_min=opt.minlr) 
    
    logger.info("==> Opts for this training: "+str(opt))

    if opt.whole_dataset:
        logger.debug("==> Using the whole dataset for training...")
        train_loader, test_loader = dataset.get_task_loaders()
    else:
        # fill the memory
        logger.debug("==> Exposing Sampler to Dataset...")
        dataset_loader, test_loader = dataset.get_task_loaders()
        for (data_unencoded, target) in dataset_loader:
            descriptor = encoder(data_unencoded.to(opt.device))
            compressed_data = compressor.compress(descriptor)
            memory.new_batch(compressed_data, target)
        train_loader = memory.get_train_loader(opt)

    logger.debug("==> Starting Train-Test Loop...")
    best_acc = 0.0
    for epoch in range(opt.num_passes):
        # Handle lr scheduling
        if epoch <= 0: # Warm start of 1 epoch
            for param_group in optimizer.param_groups:
                param_group['lr'] = opt.maxlr * 0.1
        elif epoch == 1: # Then set to maxlr
            for param_group in optimizer.param_groups:
                param_group['lr'] = opt.maxlr
        else: # Aand go!
            scheduler.step()

        # Train and test loop
        logger.info("==> Starting pass number: "+str(epoch)+", Learning rate: " + str(optimizer.param_groups[0]['lr']))
        acc = test(
            opt=opt,
            encoder=encoder,
            loader=test_loader,
            model=backbone,
            criterion=criterion,
            class_mask=dataset.class_mask,
            logger=logger,
            epoch=epoch
        )
        backbone, optimizer = train(opt=opt,
                                    loader=train_loader,
                                    model=backbone,
                                    compressor=compressor,
                                    criterion=criterion,
                                    optimizer=optimizer,
                                    epoch=epoch,
                                    logger=logger)
        
        # Log performance
        logger.info('==> Current accuracy: [{:.3f}]\t'.format(acc))
        if acc > best_acc:
            logger.info('==> Accuracies\tPrevious: [{:.3f}]\t'.format(best_acc) + 'Current: [{:.3f}]\t'.format(acc))
            best_acc = float(acc)

    logger.info('==> Training completed! Acc: [{0:.3f}]'.format(best_acc))
    return best_acc, backbone


def main():
    general_args = parse_general_args()

    add_encoder_args = get_encoder_arg_fn(general_args.encoder)
    add_compressor_args = get_compressor_arg_fn(general_args.compressor)
    add_backbone_args = get_backbone_arg_fn(general_args.backbone)
    add_sampler_args = get_sampler_arg_fn(general_args.sampler)

    parser = argparse.ArgumentParser()
    add_general_args(parser)
    add_encoder_args(parser)
    add_compressor_args(parser)
    add_backbone_args(parser)
    add_sampler_args(parser)
    opt = parser.parse_args()

    seed_everything(opt.seed)

    dataset = get_dataset(opt)
    encoder = get_encoder(opt, dataset.info())
    compressor = get_compressor(opt, encoder.info())
    backbone = get_backbone(opt, dataset.info(), compressor.info())
    memory = get_sampler(opt, encoder.info())

    console_logger = get_logger(folder=opt.log_dir+'/'+opt.exp_name+'/')
    console_logger.debug("==> Starting Continual Learning...")

    experiment(opt=opt,
               dataset=dataset,
               encoder=encoder,
               compressor=compressor,
               memory=memory,
               backbone=backbone,
               logger=console_logger)


if __name__ == '__main__':
    main()