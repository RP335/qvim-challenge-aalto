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

from qvim_mn_baseline.dataset import VimSketchDataset, AESAIMLA_DEV
from aug_dataset import AugDataset
from preprocess import AugmentMelSTFT  # Unified preprocess for both audio & spectrogram
from qvim_mn_baseline.download import (
    download_vimsketch_dataset,
    download_qvim_dev_dataset,
)
from qvim_mn_baseline.mn.model import get_model as get_mobilenet  # For MobileNet option
from qvim_mn_baseline.utils import NAME_TO_WIDTH  # For MobileNet
from qvim_mn_baseline.metrics import compute_mrr, compute_ndcg

try:
    from hear21passt.base import get_scene_embeddings as get_passt_scene_embeddings_fn
    from hear21passt.base import load_model as load_passt_hf_fn

    PASST_AVAILABLE = True
except ImportError:
    print(
        "WARNING: hear21passt library not found. PaSST model type will not be available."
    )
    PASST_AVAILABLE = False

try:
    from panns_inference import AudioTagging as PannsAudioTaggingModel

    PANNS_AVAILABLE = True
except ImportError:
    print(
        "WARNING: panns_inference library not found. PANNs model type will not be available."
    )
    PANNS_AVAILABLE = False

try:
    from speechbrain.lobes.models.beats import BEATs as SpeechBrainBEATsModel

    BEATS_AVAILABLE = True
except ImportError:
    print(
        "WARNING: speechbrain library not found (for BEATs). BEATs model type will not be available."
    )
    BEATS_AVAILABLE = False


def train_alternate(config):
    if config.random_seed is not None:
        pl.seed_everything(config.random_seed, workers=True)

    wandb_logger = WandbLogger(project=config.project, config=vars(config))

    # Initialize training dataset (standard or augmented)
    if config.use_augmentations:
        train_ds = AugDataset(
            root=os.path.join(config.dataset_path, "Vim_Sketch_Dataset"),
            sample_rate=config.sample_rate,
            duration=config.duration,
            aug_profile=config.aug_profile,
        )
    else:
        train_ds = VimSketchDataset(
            os.path.join(config.dataset_path, "Vim_Sketch_Dataset"),
            sample_rate=config.sample_rate,
            duration=config.duration,
        )

    # Validation dataset remains unchanged
    eval_ds_path = os.path.join(config.dataset_path, "DEVUpdatedDataset")
    eval_ds = AESAIMLA_DEV(
        eval_ds_path, sample_rate=config.sample_rate, duration=config.duration
    )

    train_dl = DataLoader(
        train_ds,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
    )
    eval_dl = DataLoader(
        eval_ds,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=True,
    )

    # Create a LightningDataModule (recommended for PL)
    class QVIDataModule(pl.LightningDataModule):
        def __init__(self, train_loader, val_loader):
            super().__init__()
            self._train_dataloader = train_loader
            self._val_dataloader = val_loader

        def train_dataloader(self):
            return self._train_dataloader

        def val_dataloader(self):
            return self._val_dataloader

    data_module = QVIDataModule(train_dl, eval_dl)

    pl_module = QVIMModuleAlternate(config)

    callbacks = []
    if config.model_save_path:
        callbacks.append(
            ModelCheckpoint(
                dirpath=os.path.join(
                    config.model_save_path,
                    config.model_type,
                    wandb_logger.experiment.name,
                ),
                filename="{epoch}-{val_mrr:.3f}-{val_ndcg:.3f}",
                monitor="val/mrr",
                mode="max",
                save_top_k=3,
                save_last=True,
            )
        )
    if not config.use_custom_lr_scheduler:
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    trainer = pl.Trainer(
        max_epochs=config.n_epochs,
        logger=wandb_logger,
        accelerator="auto",
        devices=(
            config.num_gpus
            if torch.cuda.is_available() and config.num_gpus > 0
            else "auto"
        ),
        callbacks=callbacks,
        deterministic=True if config.random_seed is not None else False,
    )

    print("Running initial validation...")
    trainer.validate(pl_module, datamodule=data_module)
    print("Starting training...")
    trainer.fit(pl_module, datamodule=data_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train QVIM model with alternate encoders."
    )
    # General args
    parser.add_argument(
        "--project", type=str, default="qvim-custom-models", help="W&B project name."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=min(os.cpu_count(), 8),
        help="DataLoader workers.",
    )
    parser.add_argument(
        "--num_gpus", type=int, default=1, help="Number of GPUs (0 for CPU)."
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        default="model_checkpoints_custom",
        help="Path to save checkpoints.",
    )
    parser.add_argument(
        "--dataset_path", type=str, default="data", help="Base path to datasets."
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help="Seed for reproducibility (None for random).",
    )

    # Data augmentation flags
    parser.add_argument(
        "--use_augmentations",
        action="store_true",
        help="If set, uses the augmented dataset instead of the standard one.",
    )
    parser.add_argument(
        "--aug_profile",
        type=str,
        default="light",
        choices=["light", "full"],
        help="Augmentation profile to use when --use_augmentations is set.",
    )

    # Model selection
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["mobilenet", "passt", "panns", "beats"],
        help="Encoder model.",
    )
    parser.add_argument(
        "--use_augmentations",
        action="store_true",
        help="If set, uses the augmented dataset instead of the standard one.",
    )
    parser.add_argument(
        "--aug_profile",
        type=str,
        default="light",
        choices=["light", "full"],
        help="Augmentation profile to use when --use_augmentations is set.",
    )

    # Model selection
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["mobilenet", "passt", "panns", "beats"],
        help="Encoder model.",
    )

    # --- Training & Scheduler Arguments ---
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Number of samples per batch."
    )
    parser.add_argument(
        "--n_epochs", type=int, default=50, help="Total number of training epochs."
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-5, help="L2 weight regularization."
    )
    parser.add_argument(
        "--max_lr", type=float, default=3e-4, help="Maximum learning rate."
    )
    parser.add_argument(
        "--min_lr", type=float, default=1e-5, help="Final learning rate."
    )
    parser.add_argument(
        "--warmup_epochs", type=int, default=5, help="Number of warm-up epochs."
    )
    parser.add_argument(
        "--rampdown_epochs", type=int, default=25, help="Duration for LR ramp-down."
    )
    parser.add_argument(
        "--initial_tau", type=float, default=0.07, help="Temperature for loss function."
    )
    parser.add_argument(
        "--tau_trainable",
        type=lambda x: (str(x).lower() == "true"),
        default=True,
        help="Make tau trainable.",
    )
    parser.add_argument(
        "--use_custom_lr_scheduler",
        type=lambda x: (str(x).lower() == "true"),
        default=True,
        help="Use the manual custom LR scheduler.",
    )
    parser.add_argument(
        "--target_classes",
        type=str,
        default=None,
        help="Comma-separated list of class names for MRR-based augmentation.",
    )

    # --- Preprocessing & Spectrogram Arguments ---
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Duration of audio clips in seconds.",
    )
    parser.add_argument(
        "--sample_rate", type=int, default=32000, help="Target sampling rate."
    )
    parser.add_argument(
        "--window_size", type=int, default=800, help="STFT window size in samples."
    )
    parser.add_argument(
        "--hop_size", type=int, default=320, help="STFT hop length in samples."
    )
    parser.add_argument("--n_fft", type=int, default=1024, help="Number of FFT bins.")
    parser.add_argument(
        "--n_mels", type=int, default=128, help="Number of mel filter banks."
    )
    parser.add_argument(
        "--freqm",
        type=int,
        default=48,
        help="Frequency masking parameter (SpecAugment).",
    )
    parser.add_argument(
        "--timem", type=int, default=192, help="Time masking parameter (SpecAugment)."
    )
    parser.add_argument(
        "--fmin", type=int, default=0, help="Minimum frequency for Mel spectrogram."
    )
    parser.add_argument(
        "--fmax",
        type=int,
        default=None,
        help="Maximum frequency for Mel spectrogram (None for Nyquist).",
    )
    parser.add_argument(
        "--fmin_aug_range",
        type=int,
        default=10,
        help="Variation range for fmin augmentation.",
    )
    parser.add_argument(
        "--fmax_aug_range",
        type=int,
        default=2000,
        help="Variation range for fmax augmentation.",
    )

    # --- Model-Specific Arguments ---
    # MobileNet
    parser.add_argument(
        "--pretrained_name",
        type=str,
        default="mn10_as",
        help="MobileNet pretrained model name.",
    )
    parser.add_argument(
        "--mobilenet_dummy_classes",
        type=int,
        default=527,
        help="Dummy output classes for MobileNet head.",
    )
    # PaSST
    parser.add_argument(
        "--passt_model_identifier",
        type=str,
        default="passt_s_swa_p16_128_ap476",
        help="PaSST model identifier for hear21passt.",
    )
    parser.add_argument(
        "--passt_input_type",
        type=str,
        default="raw",
        choices=["raw", "mel"],
        help="Input type for PaSST.",
    )
    # PANNs
    parser.add_argument(
        "--panns_checkpoint_path",
        type=str,
        default=None,
        help="Path to PANNs checkpoint file.",
    )
    parser.add_argument(
        "--panns_input_type",
        type=str,
        default="raw",
        choices=["raw", "mel"],
        help="Input type for PANNs.",
    )
    # BEATs
    parser.add_argument(
        "--beats_checkpoint_path",
        type=str,
        default="beats_checkpoint/BEATs_iter3.pt",
        help="Path to BEATs checkpoint.",
    )
    parser.add_argument(
        "--beats_savedir",
        type=str,
        default="pretrained_models_cache",
        help="Directory to save downloaded BEATs models.",
    )
    args = parser.parse_args()
    # Maintain existing defaults logic
    if args.fmax is None:
        args.fmax = args.sample_rate // 2
    if args.panns_checkpoint_path is None and args.model_type == "panns":
        args.panns_checkpoint_path = "/Users/rahul_peter/panns_data/Cnn14_mAP=0.431.pth"
        print(f"Using default PANNs checkpoint: {args.panns_checkpoint_path}")

    train_alternate(args)
