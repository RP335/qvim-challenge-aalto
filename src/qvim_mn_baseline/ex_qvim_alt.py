import argparse
import os
import math
import copy  # Keep this import
from copy import deepcopy

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
from qvim_mn_baseline.download import download_vimsketch_dataset, download_qvim_dev_dataset
from qvim_mn_baseline.mn.preprocess import AugmentMelSTFT
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


class QVIMModuleAlternate(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.model_type = config.model_type.lower()

        if self.model_type in ["mobilenet", "passt", "panns"] and \
                (self.model_type != "passt" or getattr(config, 'passt_input_type', 'mel') == 'mel') and \
                (self.model_type != "panns" or getattr(config, 'panns_input_type', 'raw') == 'mel'):
            self.mel = AugmentMelSTFT(
                n_mels=config.n_mels, sr=config.sample_rate, win_length=config.window_size,
                hopsize=config.hop_size, n_fft=config.n_fft, freqm=config.freqm, timem=config.timem,
                fmin=config.fmin, fmax=config.fmax, fmin_aug_range=config.fmin_aug_range,
                fmax_aug_range=config.fmax_aug_range
            )
        else:
            self.mel = None  # BEATs, raw PaSST/PANNs input use raw audio

        # Load the specified encoder
        if self.model_type == "mobilenet":
            self.imitation_encoder = get_mobilenet(
                width_mult=NAME_TO_WIDTH(config.pretrained_name),
                pretrained_name=config.pretrained_name,
                num_classes=config.mobilenet_dummy_classes,
                head_type='mlp'
            )
        elif self.model_type == "passt":
            if not PASST_AVAILABLE: raise RuntimeError("PaSST library (hear21passt) not available.")
            # load_passt_hf_fn can take `arch` (e.g. 'passt_s_p16_s12_f128_ap476') or `model_path`
            # We assume config.passt_model_identifier is the arch name or path for `load_model`
            self.imitation_encoder = load_passt_hf_fn(model_path=config.passt_model_identifier)
            # PaSST model is loaded. get_passt_scene_embeddings_fn will be used in _extract_embeddings
            # To fine-tune, ensure parameters are trainable
            for param in self.imitation_encoder.parameters():
                param.requires_grad = True
        elif self.model_type == "panns":
            if not PANNS_AVAILABLE: raise RuntimeError("PANNs library (panns_inference) not available.")
            self.imitation_encoder = PannsAudioTaggingModel(
                checkpoint_path=config.panns_checkpoint_path,
                device='cpu'  # PL will move to correct device
            )
            # Ensure PANNs is fine-tunable
            for param in self.imitation_encoder.parameters():
                param.requires_grad = True
        elif self.model_type == "beats":
            if not BEATS_AVAILABLE: raise RuntimeError("BEATs library (speechbrain) not available.")
            # `source` can be a local path to .pt file or HuggingFace identifier
            self.imitation_encoder = SpeechBrainBEATsModel(
                source=config.beats_checkpoint_path,
                savedir=os.path.join(getattr(config, 'beats_savedir', 'pretrained_models'), "beats"),
                # Optional: where to save downloaded models
                freeze=False  # Important for fine-tuning
            )
            # SpeechBrain's freeze=False should make it trainable. Double-check if needed.
            for param in self.imitation_encoder.parameters():
                param.requires_grad = True
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        self.reference_encoder = deepcopy(self.imitation_encoder)

        initial_tau = torch.zeros((1,)) + config.initial_tau
        self.tau = torch.nn.Parameter(initial_tau, requires_grad=config.tau_trainable)
        self.validation_output = []

    def _extract_embeddings(self, encoder, audio_batch):
        embedding = None
        if self.model_type == "mobilenet":
            mel_out = self.mel(audio_batch).unsqueeze(1)
            _, embedding = encoder(mel_out)
        elif self.model_type == "passt":
            # hear21passt.base.get_scene_embeddings takes (audio_waveform, model)
            # It handles the conversion to log-mel spectrograms internally if fed raw audio.
            # The shape of audio_batch from dataloader is (batch, time_samples)
            if self.config.passt_input_type == 'raw':
                # get_passt_scene_embeddings_fn expects a list of numpy arrays or a batched tensor
                # If audio_batch is already a tensor (B, T), it might work directly.
                # Otherwise, might need: [wav.cpu().numpy() for wav in audio_batch]
                # Let's assume it can handle a batch tensor.
                embedding = get_passt_scene_embeddings_fn(audio_batch, encoder)
            elif self.config.passt_input_type == 'mel':
                if self.mel is None:
                    raise ValueError("Mel module not initialized for PaSST with mel input.")
                mel_spec = self.mel(audio_batch)  # (B, F, T)
                # PaSST's internal processing might still be applied by get_scene_embeddings,
                # or the encoder itself might take melspecs if underlying model is adapted.
                # The PaSST model from `load_model` itself is callable.
                # encoder(mel_spec) -> typically gives (logits, embeddings) or just embeddings for some modes.
                # The printout shows `pre_logits` and `head`. We need output of `encoder.norm`.
                # For simplicity and standard usage, relying on get_passt_scene_embeddings_fn with raw audio is safer.
                # If you must feed mels, you need to ensure the `encoder` (PaSST model) expects them
                # and gives embeddings before the head.
                # This part might need adjustment based on how `hear21passt` expects custom mel inputs
                # if bypassing its internal preprocessing.
                # A common way is to pass it through model.forward_features() or similar.
                # For now, let's stick to what `get_passt_scene_embeddings_fn` is designed for, which is often raw audio.
                # If using the model directly: x = encoder.patch_embed(mel_spec.unsqueeze(1)) # if mel_spec is (B,F,T)
                # x = encoder.forward_blocks(x)
                # x = encoder.norm(x)
                # embedding = x.mean(dim=1) # Example global average pooling if x is sequence of patches
                raise NotImplementedError(
                    "Feeding pre-computed melspecs to PaSST via this generic path needs specific handling. Use 'raw' for passt_input_type for now.")
            else:
                raise ValueError(f"Invalid passt_input_type: {self.config.passt_input_type}")

        elif self.model_type == "panns":
            # panns_inference.AudioTagging.forward(audio) returns (clipwise_output, embedding)
            # It expects raw audio of shape (batch_size, num_samples)
            if self.config.panns_input_type == 'raw':
                _, embedding = encoder(audio_batch)
            elif self.config.panns_input_type == 'mel':
                if self.mel is None:
                    raise ValueError("Mel module not initialized for PANNs with mel input.")
                mel_spec = self.mel(audio_batch)  # (B, F, T)
                # The PANNs AudioTagging model usually has its own mel frontend.
                # Feeding custom mels requires using the core Cnn14 model directly.
                # For AudioTagging wrapper, raw input is standard.
                raise NotImplementedError(
                    "Feeding pre-computed melspecs to PANNs AudioTagging wrapper needs specific handling. Use 'raw' for panns_input_type.")
            else:
                raise ValueError(f"Invalid panns_input_type: {self.config.panns_input_type}")

        elif self.model_type == "beats":
            # SpeechBrain BEATs model's forward or extract_features takes raw audio.
            # Its output might be a tuple or a tensor.
            # `encoder.extract_features(audio_batch)` is a common way.
            output_features = encoder.extract_features(audio_batch)  # Expected: (B, Seq, Dim)

            if isinstance(output_features, tuple):  # If model returns more than features
                output_features = output_features[0]  # Assuming first element is the embedding sequence

            if output_features.ndim == 3:
                embedding = torch.mean(output_features, dim=1)  # Mean pool over sequence
            elif output_features.ndim == 2:  # Already pooled
                embedding = output_features
            else:
                raise ValueError(f"Unexpected embedding dimension from BEATs: {output_features.shape}")
        else:
            raise ValueError(f"Unsupported model type in _extract_embeddings: {self.model_type}")

        if embedding is None:
            raise RuntimeError(f"Embedding extraction failed for model type {self.model_type}")
        return torch.nn.functional.normalize(embedding, dim=1)

    def forward(self, queries, items):
        return self._extract_embeddings(self.imitation_encoder, queries), \
            self._extract_embeddings(self.reference_encoder, items)

    def training_step(self, batch, batch_idx):
        if self.config.use_custom_lr_scheduler:
            self.lr_scheduler_step_manual(batch_idx)

        y_imitation, y_reference = self(batch['imitation'], batch['reference'])

        C = torch.matmul(y_imitation, y_reference.T)
        C = C / torch.abs(self.tau)
        I = torch.eye(y_imitation.size(0), device=self.device, dtype=torch.bool)
        loss = -torch.log_softmax(C, dim=1)[I].mean()

        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/tau', self.tau.abs(), on_step=False, on_epoch=True, logger=True)
        if self.config.use_custom_lr_scheduler:
            current_lr = self.optimizers().param_groups[0]['lr']
            self.log('train/lr', current_lr, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y_imitation, y_reference = self(batch['imitation'], batch['reference'])
        self.validation_output.append({
            'imitation_emb': y_imitation.cpu().numpy(),
            'reference_emb': y_reference.cpu().numpy(),
            'imitation_filename': batch['imitation_filename'],
            'reference_filename': batch['reference_filename'],
            'imitation_class': batch['imitation_class'],
            'reference_class': batch['reference_class']
        })
        C = torch.matmul(y_imitation, y_reference.T)
        C = C / torch.abs(self.tau)
        I = torch.eye(y_imitation.size(0), device=self.device, dtype=torch.bool)
        loss = -torch.log_softmax(C, dim=1)[I].mean()
        self.log('val/loss_step', loss, on_step=False, on_epoch=True)  # Log per epoch

    def on_validation_epoch_end(self):
        if not self.validation_output:
            self.log_dict({'val/mrr': 0.0, 'val/ndcg': 0.0, 'val/loss': 0.0})
            self.validation_output.clear()  # Clear even if empty
            return

        imitation_embs = np.concatenate([x['imitation_emb'] for x in self.validation_output])
        reference_embs_all_paired = np.concatenate(
            [x['reference_emb'] for x in self.validation_output])  # These are from paired data
        imitation_filenames = sum([list(x['imitation_filename']) for x in self.validation_output], [])
        reference_filenames_paired = sum([list(x['reference_filename']) for x in self.validation_output], [])
        imitation_classes = sum([list(x['imitation_class']) for x in self.validation_output], [])
        reference_classes_all_paired = sum([list(x['reference_class']) for x in self.validation_output], [])

        # Create unique global reference set for similarity matrix
        # The DEV dataset structure provides a Query and an Item (reference) per row in DEVUpdateComplete.csv
        # The AESAIMLA_DEV loader already creates these pairs.
        # For evaluation, we compare each query's embedding against all unique *item* embeddings from the val set.

        unique_ref_embeddings_map = {}
        for i in range(len(reference_filenames_paired)):
            ref_fn = reference_filenames_paired[i]
            if ref_fn not in unique_ref_embeddings_map:
                unique_ref_embeddings_map[ref_fn] = {
                    'emb': reference_embs_all_paired[i],
                    'class': reference_classes_all_paired[i]
                }

        unique_ref_filenames_list = list(unique_ref_embeddings_map.keys())
        if not unique_ref_filenames_list:  # No references found
            self.log_dict({'val/mrr': 0.0, 'val/ndcg': 0.0})
            self.validation_output.clear()
            return

        unique_ref_embeddings_np = np.array([unique_ref_embeddings_map[fn]['emb'] for fn in unique_ref_filenames_list])
        unique_ref_classes_list = [unique_ref_embeddings_map[fn]['class'] for fn in unique_ref_filenames_list]

        scores_matrix = np.dot(imitation_embs, unique_ref_embeddings_np.T)
        similarity_df = pd.DataFrame(scores_matrix, index=imitation_filenames, columns=unique_ref_filenames_list)

        gt_mrr = {ifn: rfn_pair for ifn, rfn_pair in zip(imitation_filenames, reference_filenames_paired)}

        gt_ndcg = {}
        imitation_filename_to_class = {ifn: icls for ifn, icls in zip(imitation_filenames, imitation_classes)}

        for im_fn in imitation_filenames:
            im_cls = imitation_filename_to_class[im_fn]
            # Relevant items for NDCG are those items (references) from the unique set that share the same class as the query's class.
            gt_ndcg[im_fn] = [r_fn for r_idx, r_fn in enumerate(unique_ref_filenames_list) if
                              unique_ref_classes_list[r_idx] == im_cls]

        mrr = compute_mrr(similarity_df, gt_mrr)
        ndcg = compute_ndcg(similarity_df, gt_ndcg)

        avg_val_loss = self.trainer.callback_metrics.get('val/loss_step', torch.tensor(0.0)).item()  # Get epoch average

        self.log_dict({'val/mrr': mrr, 'val/ndcg': ndcg, 'val/loss': avg_val_loss}, prog_bar=True)
        self.validation_output.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.config.max_lr, betas=(0.9, 0.999), eps=1e-8,
            weight_decay=self.config.weight_decay, amsgrad=False
        )
        if self.config.use_custom_lr_scheduler:
            return optimizer
        else:
            if self.trainer.max_steps == -1:  # if -1, then max_epochs is used
                total_steps = self.config.n_epochs * len(self.trainer.datamodule.train_dataloader())
            else:
                total_steps = self.trainer.max_steps

            print(f"INFO: Configuring CosineAnnealingLR with T_max = {total_steps} steps.")
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
            return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    def lr_scheduler_step_manual(self, batch_idx_in_epoch):
        if self.trainer is None or not hasattr(self.trainer,
                                               'num_training_batches') or self.trainer.num_training_batches == 0:
            opt = self.optimizers(use_pl_optimizer=False)
            if opt:
                for param_group in opt.param_groups: param_group['lr'] = self.config.min_lr
            return

        steps_per_epoch = self.trainer.num_training_batches
        current_total_step = self.trainer.global_step
        min_lr, max_lr = self.config.min_lr, self.config.max_lr
        warmup_total_steps = self.config.warmup_epochs * steps_per_epoch
        rampdown_total_steps_period = (self.config.warmup_epochs + self.config.rampdown_epochs) * steps_per_epoch
        decay_duration_steps = rampdown_total_steps_period - warmup_total_steps

        lr = 0.0
        if current_total_step < warmup_total_steps:
            lr = min_lr + (max_lr - min_lr) * (
                        current_total_step / warmup_total_steps) if warmup_total_steps > 0 else max_lr
        elif current_total_step < rampdown_total_steps_period:
            decay_progress = (
                                         current_total_step - warmup_total_steps) / decay_duration_steps if decay_duration_steps > 0 else 1.0
            lr = min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * decay_progress))
        else:
            lr = min_lr

        optimizer = self.optimizers(use_pl_optimizer=False)
        if optimizer:
            for param_group in optimizer.param_groups: param_group['lr'] = lr


def train_alternate(config):
    if config.random_seed is not None:
        pl.seed_everything(config.random_seed, workers=True)

    # download_vimsketch_dataset(config.dataset_path)
    # # Corrected path for DEV dataset
    # download_qvim_dev_dataset(os.path.join(config.dataset_path,
    #                                        "DEVUpdatedDataset_Parent_Dir_Placeholder"))  # download function expects parent of 'qvim-dev'
    # but AESAIMLA_DEV expects 'DEVUpdatedDataset'
    # This might need adjustment based on download_qvim_dev_dataset behavior
    # Or ensure DEVUpdatedDataset is directly under config.dataset_path

    wandb_logger = WandbLogger(project=config.project, config=vars(config))

    train_ds = VimSketchDataset(
        os.path.join(config.dataset_path, 'Vim_Sketch_Dataset'),
        sample_rate=config.sample_rate, duration=config.duration
    )
    # Corrected path for eval_ds
    eval_ds_path = os.path.join(config.dataset_path, 'DEVUpdatedDataset')
    if not os.path.exists(os.path.join(eval_ds_path, 'DEVUpdateComplete.csv')):
        print(f"WARNING: DEVUpdateComplete.csv not found in {eval_ds_path}. Evaluation might fail.")
        # Attempt to call download if it places it correctly
        print(f"Attempting to ensure {eval_ds_path} (for AESAIMLA_DEV) is downloaded via download_qvim_dev_dataset...")
        # The download function might create a 'qvim-dev' subdir. We need 'DEVUpdatedDataset'.
        # This part is tricky if the download script has a fixed output name.
        # For now, we assume DEVUpdatedDataset exists at the specified path.
        # download_qvim_dev_dataset(config.dataset_path) # Call with base dataset path

    eval_ds = AESAIMLA_DEV(
        eval_ds_path,  # Use corrected path
        sample_rate=config.sample_rate, duration=config.duration
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
        devices=config.num_gpus if torch.cuda.is_available() and config.num_gpus > 0 else "auto",  # auto for CPU
        callbacks=callbacks,
        deterministic=True if config.random_seed is not None else False,
        # strategy="ddp_find_unused_parameters_true" if config.num_gpus > 1 else "auto" # If DDP issues
    )
    # trainer.tune(pl_module, datamodule=data_module) # For LR finder, batch size finder (optional)

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

    # Model selection
    parser.add_argument('--model_type', type=str, required=True, choices=['mobilenet', 'passt', 'panns', 'beats'],
                        help="Encoder model.")
    # MobileNet specific
    parser.add_argument('--mobilenet_dummy_classes', type=int, default=527)
    parser.add_argument('--pretrained_name', type=str, default="mn10_as")
    # PaSST specific
    parser.add_argument('--passt_model_identifier', type=str, default="passt_s_swa_p16_128_ap476",
                        help="PaSST arch name or path for hear21passt.load_model.")
    parser.add_argument('--passt_input_type', type=str, default="raw", choices=["raw", "mel"],
                        help="Input type for PaSST ('raw' recommended for get_scene_embeddings).")
    # PANNs specific
    parser.add_argument('--panns_checkpoint_path', type=str, default=None,
                        help="PANNs checkpoint path. User default: /Users/rahul_peter/panns_data/Cnn14_mAP=0.431.pth")
    parser.add_argument('--panns_input_type', type=str, default="raw", choices=["raw", "mel"],
                        help="Input type for PANNs ('raw' recommended for AudioTagging).")
    # BEATs specific
    parser.add_argument('--beats_checkpoint_path', type=str, default="beats_checkpoint/BEATs_iter3.pt",
                        help="BEATs checkpoint path or HuggingFace ID (e.g., 'microsoft/beats-iter3').")
    parser.add_argument('--beats_savedir', type=str, default="pretrained_models_cache",
                        help="Directory to save downloaded BEATs models.")

    # Training args
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--max_lr', type=float, default=3e-4)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--warmup_epochs', type=int, default=5, help="For custom LR scheduler.")
    parser.add_argument('--rampdown_epochs', type=int, default=25, help="For custom LR scheduler.")
    parser.add_argument('--initial_tau', type=float, default=0.07)
    parser.add_argument('--tau_trainable', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--use_custom_lr_scheduler', type=lambda x: (str(x).lower() == 'true'), default=True)

    # Preprocessing & Spectrogram args (used if self.mel is active)
    parser.add_argument('--duration', type=float, default=10.0)
    parser.add_argument('--sample_rate', type=int, default=32000)
    parser.add_argument('--window_size', type=int, default=800)
    parser.add_argument('--hop_size', type=int, default=320)
    parser.add_argument('--n_fft', type=int, default=1024)
    parser.add_argument('--n_mels', type=int, default=128)
    parser.add_argument('--freqm', type=int, default=48)
    parser.add_argument('--timem', type=int, default=192)
    parser.add_argument('--fmin', type=int, default=0)
    parser.add_argument('--fmax', type=int, default=None, help="Max Mel freq (None for Nyquist).")
    parser.add_argument('--fmin_aug_range', type=int, default=10)
    parser.add_argument('--fmax_aug_range', type=int, default=2000)

    args = parser.parse_args()
    if args.fmax is None: args.fmax = args.sample_rate // 2
    if args.panns_checkpoint_path is None and args.model_type == "panns":  # User specific default
        args.panns_checkpoint_path = "/Users/rahul_peter/panns_data/Cnn14_mAP=0.431.pth"
        print(f"Using default PANNs checkpoint: {args.panns_checkpoint_path}")

    train_alternate(args)