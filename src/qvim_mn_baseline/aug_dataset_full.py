# File: my_dataset.py
import re
import torch
import numpy as np
import random
from audiomentations import Compose, AddGaussianNoise, PitchShift, TimeStretch
from qvim_mn_baseline.dataset import VimSketchDataset
from audio_temporal_modifier import  TemporalModifier



class ClassTargetedAugmentedVimSketch(VimSketchDataset):
    def __init__(
            self,
            dataset_dir: str,
            sample_rate: int,
            duration: float,
            class_mrr_scores: dict = None,
            augment: bool = True,
            base_augmentation_prob: float = 0.3,
            mrr_influence_factor: float = 0.7,
            # --- New parameters for new augmentations ---
            apply_envelope_modulation_prob: float = 0.0,  # Probability for T-Foley inspired envelope mod
            apply_specmix_prob: float = 0.0,  # Probability for SpecMix (if done in __getitem__)
            specmix_gamma_range: tuple = (0.1, 0.5)  # Example: range for SpecMix's mixing ratio
    ):
        super().__init__(dataset_dir, sample_rate, duration)
        self.augment = augment
        self.class_mrr_scores = class_mrr_scores if class_mrr_scores else {}
        self.base_augmentation_prob = base_augmentation_prob
        self.mrr_influence_factor = mrr_influence_factor

        self.apply_envelope_modulation_prob = apply_envelope_modulation_prob
        self.apply_specmix_prob = apply_specmix_prob
        # (Note: SpecMix is typically better in collate_fn, but we can plan for a __getitem__ version too)

        # Pre-process class names
        self.class_info = {}
        if self.class_mrr_scores:
            for class_name, mrr_score in self.class_mrr_scores.items():
                tokens = re.findall(r'[A-Z][^A-Z]*', class_name)
                tokens = [t.lower() for t in tokens if t]
                if tokens:
                    self.class_info[class_name] = {'tokens': tokens, 'mrr': mrr_score}

        # Initialize existing pipeline
        if augment:
            self.pipeline = Compose([
                AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1.0),
                PitchShift(min_semitones=-2.0, max_semitones=2.0, p=1.0),  # Fixed parameters
                TimeStretch(min_rate=0.8, max_rate=1.2, p=1.0),
            ])
        else:
            self.pipeline = None

        # Initialize new augmentation handlers (if their probabilities are > 0)
        if self.apply_envelope_modulation_prob > 0:
            self.temporal_modifier = TemporalModifier(sample_rate=self.sample_rate)
        else:
            self.temporal_modifier = None

        if self.apply_specmix_prob > 0:
             self.spec_mixer = SpecMixer(sample_rate=self.sample_rate,
                                         gamma_range=specmix_gamma_range)
        else:
             self.spec_mixer = None
        if self.apply_envelope_modulation_prob > 0:
            from .audio_temporal_modifier import TemporalModifier
            self.temporal_modifier = TemporalModifier(sample_rate=self.sample_rate)
        else:
            self.temporal_modifier = None
        self.spec_mixer = None  # Placeholder

    def __getitem__(self, index):
        sample = super().__getitem__(index)

        if self.augment:
            fname = sample['reference_filename']
            fname_lower = fname.lower()
            # print(f"[DEBUG __getitem__] self.class_info size: {len(self.class_info)}, self.augment: {self.augment}")
            best_matching_class_name = None
            best_matching_class_info = None
            max_token_matches = 0

            if self.class_info:
                for class_name_iter, info_iter in self.class_info.items():
                    count = sum(tok in fname_lower for tok in info_iter['tokens'])
                    # print(f"[DEBUG ClassMatchLoop] File: {fname_lower}, Checking Class: {class_name_iter}, Tokens: {info_iter['tokens']}, Match Count: {count}") # This might be too verbose
                    if count > max_token_matches:
                        max_token_matches = count
                        best_matching_class_name = class_name_iter
                        best_matching_class_info = info_iter

            if best_matching_class_info and max_token_matches > 0:
                mrr_score = best_matching_class_info['mrr']
                # Calculate general augmentation probability based on MRR
                # This can be a base probability for *any* augmentation for this classes
                overall_class_aug_prob_boost = (1.0 - mrr_score) * self.mrr_influence_factor
                overall_class_aug_prob = self.base_augmentation_prob + overall_class_aug_prob_boost
                overall_class_aug_prob = max(0.0, min(1.0, overall_class_aug_prob))
                # print(
                #     f"[DEBUG AugProbCalc] File: {fname}, Matched Class: {best_matching_class_name}, MRR: {mrr_score:.4f}, Aug Prob: {overall_class_aug_prob:.4f}")
                #
                # print(
                #     f"[DEBUG] File: {fname}, Matched Class: {best_matching_class_name}, MRR: {mrr_score:.4f}, Aug Prob: {overall_class_aug_prob:.4f}")

                # 1. Apply existing audiomentations pipeline
                if self.pipeline and random.random() < overall_class_aug_prob:  # Use the calculated prob
                    im_np_for_pipeline = sample['imitation'].numpy() if isinstance(sample['imitation'],
                                                                                   torch.Tensor) else np.array(
                        sample['imitation'])
                    ref_np_for_pipeline = sample['reference'].numpy() if isinstance(sample['reference'],
                                                                                    torch.Tensor) else np.array(
                        sample['reference'])


                    try:
                        im_aug = self.pipeline(samples=im_np_for_pipeline, sample_rate=self.sample_rate)
                        ref_aug = self.pipeline(samples=ref_np_for_pipeline, sample_rate=self.sample_rate)
                        if not isinstance(im_aug, np.ndarray) or not isinstance(ref_aug, np.ndarray):
                            print(
                                f"[ERROR] Augmentation did not return NumPy arrays for {fname}! Imitation type: {type(im_aug)}, Reference type: {type(ref_aug)}. Skipping torch.from_numpy for this sample's augmentations.")

                        elif len(im_aug) == 0 or len(ref_aug) == 0:
                            print(
                                f"[ERROR] Augmentation resulted in empty array for {fname}! Imitation length: {len(im_aug)}, Reference length: {len(ref_aug)}. Skipping torch.from_numpy.")
                        else:
                            sample['imitation'] = torch.from_numpy(im_aug.astype(np.float32))  # Ensure float32
                            sample['reference'] = torch.from_numpy(ref_aug.astype(np.float32))  # Ensure float32

                    except Exception as e:
                        print(f"[ERROR] Exception during audiomentations for file {fname}: {e}")

                if self.temporal_modifier and random.random() < self.apply_envelope_modulation_prob:
                    sample = self.temporal_modifier.apply_envelope_modulation(
                        sample,
                        imitation_wav=sample['imitation'].numpy(),
                        reference_wav=sample['reference'].numpy()
                    )

                # 3. Apply SpecMix (if doing a __getitem__ version)
                # This is more complex as it needs a second sample.
                if self.spec_mixer and random.random() < self.apply_specmix_prob:
                    idx2 = random.randint(0, len(self) - 1)
                    while idx2 == index: # ensure different sample
                        idx2 = random.randint(0, len(self) - 1)
                    sample2 = super().__getitem__(idx2)
                    sample = self.spec_mixer.apply_item_level_specmix(sample, sample2)
                    pass # Placeholder

        if isinstance(sample['imitation'], np.ndarray):
            sample['imitation'] = torch.from_numpy(sample['imitation'])
        if isinstance(sample['reference'], np.ndarray):
            sample['reference'] = torch.from_numpy(sample['reference'])

        return sample