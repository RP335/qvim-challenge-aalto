# File: my_dataset.py
import re
import torch
import numpy as np
import random  # Added for probability check
from audiomentations import Compose, AddGaussianNoise, PitchShift, TimeStretch
from qvim_mn_baseline.dataset import \
    VimSketchDataset  # Assuming AESAIMLA_DEV is not directly used here but available via qvim_mn_baseline.dataset


class ClassTargetedAugmentedVimSketch(VimSketchDataset):
    def __init__(
            self,
            dataset_dir: str,
            sample_rate: int,
            duration: float,
            class_mrr_scores: dict = None,  # e.g., {"Doorbell": 0.14, "Birds": 0.28}
            augment: bool = True,
            base_augmentation_prob: float = 0.3,
            # Base probability of augmenting if class matches (applied when MRR is 1.0 or class not in MRR dict)
            mrr_influence_factor: float = 0.7
            # Max additional probability due to low MRR (e.g., MRR=0 adds this much, up to 1.0 total)
    ):
        super().__init__(dataset_dir, sample_rate, duration)
        self.augment = augment
        self.class_mrr_scores = class_mrr_scores if class_mrr_scores else {}
        self.base_augmentation_prob = base_augmentation_prob
        self.mrr_influence_factor = mrr_influence_factor

        # Pre-process class names into tokens for matching and store MRR
        self.class_info = {}
        for class_name, mrr_score in self.class_mrr_scores.items():
            tokens = re.findall(r'[A-Z][^A-Z]*', class_name)
            tokens = [t.lower() for t in tokens if t]
            if tokens:
                self.class_info[class_name] = {'tokens': tokens, 'mrr': mrr_score}


        if augment and self.class_info:
            self.pipeline = Compose([
                AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1.0),
                PitchShift(min_semitones=-2, max_semitones=2, p=1.0),
                TimeStretch(min_rate=0.8, max_rate=1.2, p=1.0),
            ])
        else:
            self.pipeline = None

    def __getitem__(self, index):
        # sample is a dictionary, where 'imitation' and 'reference' are numpy arrays
        sample = super().__getitem__(index)

        if self.pipeline and self.augment:
            fname = sample['reference_filename']  # e.g. '010_009Animal_…_Cluck.wav'
            fname_lower = fname.lower()
            # --- New logic: pick class with max token match ---
            best_class = None
            best_info = None
            best_count = 0

            for class_name, info in self.class_info.items():
                count = sum(tok in fname_lower for tok in info['tokens'])
                if count > best_count:
                    best_class = class_name
                    best_info = info
                    best_count = count

                if best_count > 0:
                    mrr_score = best_info['mrr']
                    additional_prob_boost = (1.0 - mrr_score) * self.mrr_influence_factor
                    current_augmentation_prob = self.base_augmentation_prob + additional_prob_boost
                    current_augmentation_prob = max(0.0, min(1.0, current_augmentation_prob))

                    if random.random() < current_augmentation_prob:
                        im_np = sample['imitation']
                        ref_np = sample['reference']

                        if not isinstance(im_np, np.ndarray):
                            im_np = im_np.numpy() if isinstance(im_np, torch.Tensor) else np.array(im_np)
                        if not isinstance(ref_np, np.ndarray):
                            ref_np = ref_np.numpy() if isinstance(ref_np, torch.Tensor) else np.array(ref_np)

                        im_aug = self.pipeline(samples=im_np, sample_rate=self.sample_rate)
                        ref_aug = self.pipeline(samples=ref_np, sample_rate=self.sample_rate)

                        sample['imitation'] = torch.from_numpy(im_aug)
                        sample['reference'] = torch.from_numpy(ref_aug)

                    break  # Augment based on the first matched class and then stop checking other classes

        # Ensure the output is tensors if not augmented, or already converted if augmented
        if isinstance(sample['imitation'], np.ndarray):
            sample['imitation'] = torch.from_numpy(sample['imitation'])
        if isinstance(sample['reference'], np.ndarray):
            sample['reference'] = torch.from_numpy(sample['reference'])

        return sample