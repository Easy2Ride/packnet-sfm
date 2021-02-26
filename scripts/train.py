# Copyright 2020 Toyota Research Institute.  All rights reserved.

import argparse
import os

from packnet_sfm.models.model_wrapper import ModelWrapper
from packnet_sfm.models.model_checkpoint import ModelCheckpoint
# 
from packnet_sfm.trainers.pytorch_trainer import PytorchTrainer
from packnet_sfm.utils.config import parse_train_file
from packnet_sfm.utils.load import set_debug, filter_args_create
from packnet_sfm.utils.horovod import hvd_init, rank

try:
    import horovod.torch as hvd
    HAS_HOROVOD = True
except ImportError:
    HAS_HOROVOD = False
# from packnet_sfm.loggers import WandbLogger


def parse_args():
    """Parse arguments for training script"""
    parser = argparse.ArgumentParser(description='PackNet-SfM training script')
    parser.add_argument('file', type=str, help='Input file (.ckpt or .yaml)')
    args = parser.parse_args()
    assert args.file.endswith(('.ckpt', '.yaml')), \
        'You need to provide a .ckpt of .yaml file'
    return args


def train(file):
    """
    Monocular depth estimation training script.

    Parameters
    ----------
    file : str
        Filepath, can be either a
        **.yaml** for a yacs configuration file or a
        **.ckpt** for a pre-trained checkpoint file.
    """
    # Produce configuration and checkpoint from filename
    config, ckpt = parse_train_file(file)
    # config.arch.max_epochs=50

    # Initialize horovod
    if hasattr(config,"gpu"):
        os.environ["CUDA_VISIBLE_DEVICES"] =  ",".join(str(x) for x in config.gpu) # "0,1"
    hvd_init()

    # Set debug if requested
    set_debug(config.debug)

    # Wandb Logger
    logger = None# if config.wandb.dry_run or rank() > 0 \
        # else filter_args_create(WandbLogger, config.wandb)

    # model checkpoint
    checkpoint = None if config.checkpoint.filepath is '' or rank() > 0 else \
        filter_args_create(ModelCheckpoint, config.checkpoint)

    # Initialize model wrapper
    model_wrapper = ModelWrapper(config, resume=ckpt, logger=logger, use_horovod=HAS_HOROVOD)

    # Create trainer with args.arch parameters
    if HAS_HOROVOD:
        from packnet_sfm.trainers.horovod_trainer import HorovodTrainer
        trainer = HorovodTrainer(checkpoint=checkpoint, **config.arch)
    else:
        trainer = PytorchTrainer(checkpoint=checkpoint, **config.arch)

    # Train model
    trainer.fit(model_wrapper)


if __name__ == '__main__':
    args = parse_args()
    train(args.file)
