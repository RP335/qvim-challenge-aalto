import argparse
import os
import math
import copy  # Keep this import
from copy import deepcopy
import pickle

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# Imports from your existing project
from qvim_mn_baseline.dataset import VimSketchDataset, AESAIMLA_DEV
from augmented_dataset import AugDataset  # New unified augmentation dataset
from preprocess import AugmentMelSTFT  # Unified preprocess for both audio & spectrogram
from qvim_mn_baseline.download import download_vimsketch_dataset, download_qvim_dev_dataset
from qvim_mn_baseline.mn.model import get_model as get_mobilenet  # For MobileNet option
from qvim_mn_baseline.utils import NAME_TO_WIDTH  # For MobileNet
from qvim_mn_baseline.metrics import compute_mrr, compute_ndcg

# Specific model imports based on user feedback
try:
    from hear21passt.base import get_scene_embeddings as get_passt_scene_embeddings_fn
    from hear21passt.base import load_model as load_passt_hf_fn

    PASST_AVAILABLE = True
except ImportError:
    print("WARNING: hear21passt library not found. PaSST model type will not be available.")
    PASST_AVAILABLE = False

try:
    from panns_inference import AudioTagging as PannsAudioTaggingModel

    PANNS_AVAILABLE = True
except ImportError:
    print("WARNING: panns_inference library not found. PANNs model type will not be available.")
    PANNS_AVAILABLE = False

try:
    from speechbrain.lobes.models.beats import BEATs as SpeechBrainBEATsModel

    BEATS_AVAILABLE = True
except ImportError:
    print("WARNING: speechbrain library not found (for BEATs). BEATs model type will not be available.")
    BEATS_AVAILABLE = False


def train_alternate(config):
    if config.random_seed is not None:
        pl.seed_everything(config.random_seed, workers=True)

    wandb_logger = WandbLogger(project=config.project, config=vars(config))

    # Initialize training dataset (standard or augmented)
    if config.use_augmentations:
        train_ds = AugDataset(
            root=os.path.join(config.dataset_path, 'Vim_Sketch_Dataset'),
            sample_rate=config.sample_rate,
            duration=config.duration,
            aug_profile=config.aug_profile
        )
    else:
        train_ds = VimSketchDataset(
            os.path.join(config.dataset_path, 'Vim_Sketch_Dataset'),
            sample_rate=config.sample_rate,
            duration=config.duration
        )

    # Validation dataset remains unchanged
    eval_ds_path = os.path.join(config.dataset_path, 'DEVUpdatedDataset')
    eval_ds = AESAIMLA_DEV(
        eval_ds_path,
        sample_rate=config.sample_rate,
        duration=config.duration
    )

    train_dl = DataLoader(train_ds, num_workers=config.num_workers, batch_size=config.batch_size, shuffle=True,
                          pin_memory=True)
    eval_dl = DataLoader(eval_ds, num_workers=config.num_workers, batch_size=config.batch_size, shuffle=False,
                         pin_memory=True)

    # Create a LightningDataModule (recommended for PL)
    class QVIDataModule(pl.LightningDataModule):
        def __init__(self, train_loader, val_loader):
            super().__init__()
            self._train_dataloader = train_loader
            self._val_dataloader = val_loader

        def train_dataloader(self): return self._train_dataloader

        def val_dataloader(self): return self._val_dataloader

    data_module = QVIDataModule(train_dl, eval_dl)

    pl_module = QVIMModuleAlternate(config)

    callbacks = []
    if config.model_save_path:
        callbacks.append(ModelCheckpoint(
            dirpath=os.path.join(config.model_save_path, config.model_type, wandb_logger.experiment.name),
            filename="{epoch}-{val_mrr:.3f}-{val_ndcg:.3f}", monitor="val/mrr", mode="max",
            save_top_k=3, save_last=True
        ))
    if not config.use_custom_lr_scheduler:
        callbacks.append(LearningRateMonitor(logging_interval='step'))

    trainer = pl.Trainer(
        max_epochs=config.n_epochs, logger=wandb_logger, accelerator='auto',
        devices=config.num_gpus if torch.cuda.is_available() and config.num_gpus > 0 else "auto",
        callbacks=callbacks,
        deterministic=True if config.random_seed is not None else False,
    )

    print("Running initial validation...")
    trainer.validate(pl_module, datamodule=data_module)
    print("Starting training...")
    trainer.fit(pl_module, datamodule=data_module)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train QVIM model with alternate encoders.")
    # General args
    parser.add_argument('--project', type=str, default="qvim-custom-models", help="W&B project name.")
    parser.add_argument('--num_workers', type=int, default=min(os.cpu_count(), 8), help="DataLoader workers.")
    parser.add_argument('--num_gpus', type=int, default=1, help="Number of GPUs (0 for CPU).")
    parser.add_argument('--model_save_path', type=str, default="model_checkpoints_custom",
                        help="Path to save checkpoints.")
    parser.add_argument('--dataset_path', type=str, default='data', help="Base path to datasets.")
    parser.add_argument('--random_seed', type=int, default=None, help="Seed for reproducibility (None for random).")

    # Data augmentation flags
    parser.add_argument('--use_augmentations', action='store_true',
                        help="If set, uses the augmented dataset instead of the standard one.")
    parser.add_argument('--aug_profile', type=str, default='light', choices=['light', 'full'],
                        help="Augmentation profile to use when --use_augmentations is set.")

    # Model selection
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['mobilenet', 'passt', 'panns', 'beats'],
                        help="Encoder model.")
    # ... (rest of your existing args remain unchanged) ...

    args = parser.parse_args()
    # Maintain existing defaults logic
    if args.fmax is None: args.fmax = args.sample_rate // 2
    if args.panns_checkpoint_path is None and args.model_type == "panns":
        args.panns_checkpoint_path = "/Users/rahul_peter/panns_data/Cnn14_mAP=0.431.pth"
        print(f"Using default PANNs checkpoint: {args.panns_checkpoint_path}")

    train_alternate(args)
