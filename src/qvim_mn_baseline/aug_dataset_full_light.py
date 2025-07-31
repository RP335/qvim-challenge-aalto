# File: aug_dataset_full_light.py
import re
import torch
import numpy as np
import random
import traceback
from audiomentations import Compose, AddGaussianNoise, PitchShift, TimeStretch
from qvim_mn_baseline.dataset import VimSketchDataset
from audio_temporal_modifier import TemporalModifier
from spec_mixer import SpecMixerSpectrogramOut  


class ClassTargetedAugmentedVimSketchLight(VimSketchDataset):
    def __init__(
        self,
        dataset_dir: str,
        sample_rate: int,
        duration: float,
        class_mrr_scores: dict = None,
        augment: bool = True,
        base_augmentation_prob: float = 0.13,
        mrr_influence_factor: float = 0.6,
        apply_envelope_modulation_prob: float = 0.05,
        apply_specmix_prob: float = 0.05,
        specmix_gamma_range: tuple = (0.1, 0.4),
        specmix_max_bands: int = 10,
        specmix_n_fft: int = 1024,
        specmix_hop_length: int = 256,
        specmix_n_mels: int = 128,
        specmix_device: str = 'cpu'
    ):
        super().__init__(dataset_dir, sample_rate, duration)
        self.augment = augment
        self.class_mrr_scores = class_mrr_scores or {}
        self.base_augmentation_prob = base_augmentation_prob
        self.mrr_influence_factor = mrr_influence_factor

        # Original audiomentations pipeline for audio
        if self.augment:
            self.audio_pipeline = Compose([
                AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.3),
                PitchShift(min_semitones=-0.5, max_semitones=0.5, p=0.3),
                TimeStretch(min_rate=0.95, max_rate=1.05, p=0.3),
            ])
        else:
            self.audio_pipeline = None

        self.apply_envelope_modulation_prob = apply_envelope_modulation_prob
        self.temporal_modifier = TemporalModifier(sample_rate=self.sample_rate) if apply_envelope_modulation_prob > 0 else None

        # SpecMix for spectrogram mixing
        self.apply_specmix_prob = apply_specmix_prob
        self.spec_mixer = (
            SpecMixerSpectrogramOut(
                sample_rate=self.sample_rate,
                n_fft=specmix_n_fft,
                hop_length=specmix_hop_length,
                n_mels=specmix_n_mels,
                gamma_range=specmix_gamma_range,
                max_time_bands=specmix_max_bands,
                max_freq_bands=specmix_max_bands,
                mix_target_device=specmix_device
            )
            if apply_specmix_prob > 0 else None
        )

        # Prepare class info for MRR-based probability
        self.class_info = {}
        for class_name, mrr_score in self.class_mrr_scores.items():
            tokens = re.findall(r'[A-Z][^A-Z]*', class_name)
            tokens = [t.lower() for t in tokens if t]
            if tokens:
                self.class_info[class_name] = {'tokens': tokens, 'mrr': mrr_score}

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        fname = sample['reference_filename']

        # Convert raw audio arrays to tensors
        imitation = torch.from_numpy(sample['imitation'].astype(np.float32))
        reference = torch.from_numpy(sample['reference'].astype(np.float32))

        if self.augment:
            # Determine augmentation probability based on MRR tokens
            fname_lower = fname.lower()
            best_info = None
            max_matches = 0
            for info in self.class_info.values():
                matches = sum(tok in fname_lower for tok in info['tokens'])
                if matches > max_matches:
                    max_matches = matches
                    best_info = info
            aug_prob = self.base_augmentation_prob
            if best_info:
                aug_prob = aug_prob + (1.0 - best_info['mrr']) * self.mrr_influence_factor
                aug_prob = min(max(aug_prob, 0.0), 1.0)

            # Apply audiomentations
            if self.audio_pipeline and random.random() < aug_prob:
                try:
                    im_np = imitation.numpy()
                    ref_np = reference.numpy()
                    im_aug = self.audio_pipeline(samples=im_np, sample_rate=self.sample_rate)
                    ref_aug = self.audio_pipeline(samples=ref_np, sample_rate=self.sample_rate)
                    imitation = torch.from_numpy(im_aug.astype(np.float32))
                    reference = torch.from_numpy(ref_aug.astype(np.float32))
                except Exception as e:
                    print(f"[ERROR Audiomentations] File: {fname}: {e}")

            # Apply temporal modulation
            if self.temporal_modifier and random.random() < self.apply_envelope_modulation_prob:
                try:
                    mod = self.temporal_modifier.apply_envelope_modulation({
                        'imitation': imitation.numpy(),
                        'reference': reference.numpy()
                    })
                    imitation =  torch.as_tensor(mod['imitation'], dtype=torch.float32)
                    reference = torch.as_tensor(mod['reference'], dtype=torch.float32)
                except Exception as e:
                    print(f"[ERROR TemporalModifier] File: {fname}: {e}")

        # SpecMix (spectrogram output)
        if self.spec_mixer and random.random() < self.apply_specmix_prob:
            audio_im_before = imitation.clone()
            audio_ref_before = reference.clone()
            try:
                idx2 = random.randint(0, len(self) - 1)
                while idx2 == index:
                    idx2 = random.randint(0, len(self) - 1)
                sample2 = super().__getitem__(idx2)
                proc_ref = self.spec_mixer.apply_item_level_specmix(
                    {'reference': audio_ref_before}, sample2, target_key='reference'
                )
                reference = proc_ref['reference']
                proc_im = self.spec_mixer.apply_item_level_specmix(
                    {'imitation': audio_im_before}, sample2, target_key='imitation'
                )
                imitation = proc_im['imitation']
            except Exception as e:
                print(f"[ERROR SpecMix] File: {fname}, idx2: {idx2}: {e}")
                imitation = audio_im_before
                reference = audio_ref_before

        # Ensure spectrogram consistency if SpecMix is active
        if self.spec_mixer:
            # Convert any leftover audio to spectrogram
            if imitation.ndim == 1:
                try:
                    spec = self.spec_mixer._waveform_to_spectrogram(
                        self.spec_mixer._to_tensor_and_device(imitation.cpu())
                    ).squeeze(0).cpu()
                    imitation = spec
                except Exception:
                    pass
            if reference.ndim == 1:
                try:
                    spec = self.spec_mixer._waveform_to_spectrogram(
                        self.spec_mixer._to_tensor_and_device(reference.cpu())
                    ).squeeze(0).cpu()
                    reference = spec
                except Exception:
                    pass

        sample['imitation'] = imitation
        sample['reference'] = reference
        return sample
