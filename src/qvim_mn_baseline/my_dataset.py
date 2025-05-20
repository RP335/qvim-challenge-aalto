# File: my_dataset.py
import re
import torch
from audiomentations import Compose, AddGaussianNoise, PitchShift, TimeStretch
from qvim_mn_baseline.dataset import VimSketchDataset, AESAIMLA_DEV
class ClassTargetedAugmentedVimSketch(VimSketchDataset):
    def __init__(
        self,
        dataset_dir: str,
        sample_rate: int,
        duration: float,
        target_classes: list[str],
        augment: bool = True
    ):
        super().__init__(dataset_dir, sample_rate, duration)
        self.augment = augment

        # Build a list of token lists, one per target class
        self.token_lists: list[list[str]] = []
        for cls in target_classes:
            # Split CamelCase into tokens
            tokens = re.findall(r'[A-Z][^A-Z]*', cls)
            tokens = [t.lower() for t in tokens if t]
            if tokens:
                self.token_lists.append(tokens)

        # Define your waveform‐level pipeline (using audiomentations)
        if augment and self.token_lists:
            self.pipeline = Compose([
                AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
                PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
                TimeStretch(min_rate=0.8, max_rate=1.2, p=0.5),
            ])
        else:
            self.pipeline = None

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        if self.pipeline:
            fname = sample['reference_filename']  # e.g. '010_009Animal_…_Cluck.wav'
            fname_lower = fname.lower()
            # if any of our target token-sets fully match, augment
            for tokens in self.token_lists:
                if all(tok in fname_lower for tok in tokens):
                    # apply to both imitation & reference
                    im_np = sample['imitation'].numpy()
                    ref_np = sample['reference'].numpy()
                    im_aug = self.pipeline(samples=im_np, sample_rate=self.sample_rate)
                    ref_aug = self.pipeline(samples=ref_np, sample_rate=self.sample_rate)
                    sample['imitation'] = torch.from_numpy(im_aug)
                    sample['reference'] = torch.from_numpy(ref_aug)
                    break
        return sample
