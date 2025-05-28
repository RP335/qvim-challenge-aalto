# aug_dataset_full_2.py
import re
import torch
import numpy as np
import random
from torch_audiomentations import Compose as TorchCompose, Gain  # Using only Gain for simplicity now
from audiomentations import Compose as NumpyCompose, PitchShift as NumpyPitchShift, TimeStretch as NumpyTimeStretch, \
    AddGaussianNoise as NumpyAddGaussianNoise
from qvim_mn_baseline.dataset import VimSketchDataset
from audio_temporal_modifier import TemporalModifier
from audio_spec_mixer_from_spectrogram import SpecMixerSpectrogramOut


class ClassTargetedAugmentedVimSketch2(VimSketchDataset):
    def __init__(
            self,
            dataset_dir: str,
            sample_rate: int,
            duration: float,
            class_mrr_scores: dict = None,
            augment: bool = True,
            base_augmentation_prob: float = 0.3,
            mrr_influence_factor: float = 0.7,

            ta_gain_min_db: float = -12.0,
            ta_gain_max_db: float = 6.0,
            ta_gain_prob: float = 0.5,

            apply_envelope_modulation_prob: float = 0.1,

            apply_specmix_prob: float = 0.3,
            specmix_gamma_range: tuple = (0.1, 0.4),
            specmix_max_bands: int = 2,
            specmix_n_fft: int = 1024,
            specmix_hop_length: int = 512,
            specmix_n_mels: int = 128,
            specmix_device: str = 'cpu'
    ):
        super().__init__(dataset_dir, sample_rate, duration)
        self.augment = augment
        self.class_mrr_scores = class_mrr_scores if class_mrr_scores else {}
        self.base_augmentation_prob = base_augmentation_prob
        self.mrr_influence_factor = mrr_influence_factor
        self.target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.apply_envelope_modulation_prob = apply_envelope_modulation_prob
        self.apply_specmix_prob = apply_specmix_prob

        self.class_info = {}
        if self.class_mrr_scores:
            for class_name, mrr_score in self.class_mrr_scores.items():
                tokens = re.findall(r'[A-Z][^A-Z]*', class_name)
                tokens = [t.lower() for t in tokens if t]
                if tokens:
                    self.class_info[class_name] = {'tokens': tokens, 'mrr': mrr_score}

        if augment:
            self.numpy_audio_pipeline = NumpyCompose([
                NumpyAddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1.0),
                NumpyPitchShift(min_semitones=-2, max_semitones=2, p=1.0),
                NumpyTimeStretch(min_rate=0.8, max_rate=1.2, p=1.0),
            ])
            self.torch_audio_pipeline = TorchCompose(
                transforms=[
                    Gain(min_gain_in_db=ta_gain_min_db, max_gain_in_db=ta_gain_max_db, p=ta_gain_prob,
                         output_type="dict")
                ],
                output_type="dict"
            ).to(self.target_device)

            if self.apply_envelope_modulation_prob > 0:
                self.temporal_modifier = TemporalModifier(sample_rate=self.sample_rate)
            else:
                self.temporal_modifier = None

            if self.apply_specmix_prob > 0:
                self.spec_mixer = SpecMixerSpectrogramOut(
                    sample_rate=self.sample_rate,
                    n_fft=specmix_n_fft,
                    hop_length=specmix_hop_length,
                    # n_mels=specmix_n_mels, # n_mels is not used by SpecMixerSpectrogramOut
                    gamma_range=specmix_gamma_range,
                    max_time_bands=specmix_max_bands,
                    max_freq_bands=specmix_max_bands,
                    mix_target_device=specmix_device
                )
            else:
                self.spec_mixer = None
        else:
            self.numpy_audio_pipeline = None
            self.torch_audio_pipeline = None
            self.temporal_modifier = None
            self.spec_mixer = None

    def __getitem__(self, index):
        sample = super().__getitem__(index)

        imitation_audio = torch.from_numpy(sample['imitation'].astype(np.float32))  # Target shape: (L,)
        reference_audio = torch.from_numpy(sample['reference'].astype(np.float32))  # Target shape: (L,)

        if self.augment:
            fname = sample['reference_filename']
            fname_lower = fname.lower()
            # print(f"[DEBUG __getitem__] File: {fname}, Start shapes: im={imitation_audio.shape}, ref={reference_audio.shape}")

            best_matching_class_name = None
            best_matching_class_info = None
            max_token_matches = 0

            if self.class_info:
                for class_name_iter, info_iter in self.class_info.items():
                    count = sum(tok in fname_lower for tok in info_iter['tokens'])
                    if count > max_token_matches:
                        max_token_matches = count
                        best_matching_class_name = class_name_iter
                        best_matching_class_info = info_iter
                # if best_matching_class_name:
                #     print(f"[DEBUG ClassMatchResult] File: {fname}, Best Match: {best_matching_class_name}, Tokens: {max_token_matches}")

            current_imitation_audio_for_aug = imitation_audio.clone()
            current_reference_audio_for_aug = reference_audio.clone()
            augmented_this_sample = False

            if best_matching_class_info and max_token_matches > 0:
                mrr_score = best_matching_class_info['mrr']
                overall_class_aug_prob = self.base_augmentation_prob + (1.0 - mrr_score) * self.mrr_influence_factor
                overall_class_aug_prob = max(0.0, min(1.0, overall_class_aug_prob))
                # print(f"[DEBUG AugProbCalc] File: {fname}, Matched: {best_matching_class_name}, MRR: {mrr_score:.4f}, Prob: {overall_class_aug_prob:.4f}")

                if random.random() < overall_class_aug_prob:
                    augmented_this_sample = True
                    # print(f"[DEBUG AugDecision] File: {fname} - WILL BE CLASS-TARGETED AUGMENTED.")

                    if self.numpy_audio_pipeline:
                        try:
                            im_np = current_imitation_audio_for_aug.cpu().numpy()
                            ref_np = current_reference_audio_for_aug.cpu().numpy()
                            im_aug_np = self.numpy_audio_pipeline(samples=im_np, sample_rate=self.sample_rate)
                            ref_aug_np = self.numpy_audio_pipeline(samples=ref_np, sample_rate=self.sample_rate)
                            current_imitation_audio_for_aug = torch.from_numpy(im_aug_np.astype(np.float32))
                            current_reference_audio_for_aug = torch.from_numpy(ref_aug_np.astype(np.float32))
                            print(f"[DEBUG NumpyAug Applied] File: {fname}, Shapes: im={current_imitation_audio_for_aug.shape}, ref={current_reference_audio_for_aug.shape}")
                        except Exception as e:
                            print(f"[ERROR NumpyAug] {fname}: {e}")

                    if self.torch_audio_pipeline:
                        try:
                            im_in = current_imitation_audio_for_aug.unsqueeze(0).unsqueeze(0).to(self.target_device)
                            ref_in = current_reference_audio_for_aug.unsqueeze(0).unsqueeze(0).to(self.target_device)
                            current_imitation_audio_for_aug = \
                            self.torch_audio_pipeline(samples=im_in, sample_rate=self.sample_rate)['samples'].squeeze(
                                0).squeeze(0).cpu()
                            current_reference_audio_for_aug = \
                            self.torch_audio_pipeline(samples=ref_in, sample_rate=self.sample_rate)['samples'].squeeze(
                                0).squeeze(0).cpu()
                            print(f"[DEBUG TorchAug Applied] File: {fname}, Shapes: im={current_imitation_audio_for_aug.shape}, ref={current_reference_audio_for_aug.shape}")
                        except Exception as e:
                            print(f"[ERROR TorchAug] {fname}: {e}")

                    if self.temporal_modifier and random.random() < self.apply_envelope_modulation_prob:
                        # print(f"[DEBUG TemporalMod Attempt] File: {fname}")
                        temp_sample_for_tm = {'imitation': current_imitation_audio_for_aug.cpu().numpy(),
                                              'reference': current_reference_audio_for_aug.cpu().numpy()}
                        try:
                            mod_tm = self.temporal_modifier.apply_envelope_modulation(temp_sample_for_tm)
                            # current_imitation_audio_for_aug = torch.from_numpy(mod_tm['imitation'].astype(np.float32))
                            current_imitation_audio_for_aug = torch.as_tensor(mod_tm['imitation'], dtype=torch.float32)
                            # current_reference_audio_for_aug = torch.from_numpy(mod_tm['reference'].astype(np.float32))
                            current_reference_audio_for_aug = torch.as_tensor(mod_tm['reference'], dtype=torch.float32)
                            print(f"[DEBUG TemporalMod Applied] File: {fname}, Shapes: im={current_imitation_audio_for_aug.shape}, ref={current_reference_audio_for_aug.shape}")
                        except Exception as e:
                            print(f"[ERROR TemporalMod] {fname}: {e}")
                # else:
                # print(f"[DEBUG AugDecision] File: {fname} - SKIPPED CLASS-TARGETED AUG (prob fail).")

            elif self.augment:  # No specific class match OR self.class_info empty, but global augment is ON
                if random.random() < self.base_augmentation_prob:
                    augmented_this_sample = True
                    # print(f"[DEBUG AugDecision General] File: {fname} - APPLYING GENERAL AUGMENTATIONS.")
                    if self.numpy_audio_pipeline:
                        try:
                            im_np = current_imitation_audio_for_aug.cpu().numpy()
                            ref_np = current_reference_audio_for_aug.cpu().numpy()
                            im_aug_np = self.numpy_audio_pipeline(samples=im_np, sample_rate=self.sample_rate)
                            ref_aug_np = self.numpy_audio_pipeline(samples=ref_np, sample_rate=self.sample_rate)
                            current_imitation_audio_for_aug = torch.from_numpy(im_aug_np.astype(np.float32))
                            current_reference_audio_for_aug = torch.from_numpy(ref_aug_np.astype(np.float32))
                            print(f"[DEBUG NumpyAug General Applied] File: {fname}")
                        except Exception as e:
                            print(f"[ERROR NumpyAug General] {fname}: {e}")

                    if self.torch_audio_pipeline:
                        try:
                            im_in = current_imitation_audio_for_aug.unsqueeze(0).unsqueeze(0).to(self.target_device)
                            ref_in = current_reference_audio_for_aug.unsqueeze(0).unsqueeze(0).to(self.target_device)
                            current_imitation_audio_for_aug = \
                            self.torch_audio_pipeline(samples=im_in, sample_rate=self.sample_rate)['samples'].squeeze(
                                0).squeeze(0).cpu()
                            current_reference_audio_for_aug = \
                            self.torch_audio_pipeline(samples=ref_in, sample_rate=self.sample_rate)['samples'].squeeze(
                                0).squeeze(0).cpu()
                            print(f"[DEBUG TorchAug General Applied] File: {fname}")
                        except Exception as e:
                            print(f"[ERROR TorchAug General] {fname}: {e}")

            if augmented_this_sample:
                imitation_audio = current_imitation_audio_for_aug
                reference_audio = current_reference_audio_for_aug

            original_imitation_audio = imitation_data.clone()
            original_reference_audio = reference_data.clone()
            if self.spec_mixer and random.random() < self.apply_specmix_prob:
                try:
                    idx2 = random.randint(0, len(self) - 1)
                    while idx2 == index: idx2 = random.randint(0, len(self) - 1)
                    sample2_raw_audio_dict = super().__getitem__(idx2)  # This is a dict of numpy arrays

                    # Process 'reference' audio
                    sample1_for_ref_mix = {'reference': original_reference_audio.clone()}  # Pass only the target audio
                    mixed_output_ref = self.spec_mixer.apply_item_level_specmix(
                        sample1_for_ref_mix,
                        sample2_raw_audio_dict,  # Contains numpy audio for sample2
                        target_key='reference'
                    )
                    reference_data = mixed_output_ref['reference']  # This is now a spectrogram (F,T) from mixer
                    reference_is_audio = False  # Mark as spectrogram

                    # Process 'imitation' audio
                    sample1_for_im_mix = {'imitation': original_imitation_audio.clone()}  # Pass only the target audio
                    mixed_output_im = self.spec_mixer.apply_item_level_specmix(
                        sample1_for_im_mix,
                        sample2_raw_audio_dict,  # Contains numpy audio for sample2
                        target_key='imitation'
                    )
                    imitation_data = mixed_output_im['imitation']  # This is now a spectrogram (F,T) from mixer
                    imitation_is_audio = False  # Mark as spectrogram

                    # print(f"[DEBUG SpecMix Applied (Outputting Spectrogram)] File: {fname}, Shapes: im={imitation_data.shape}, ref={reference_data.shape}")

                except Exception as e:
                    idx2_str = str(idx2) if 'idx2' in locals() else "UNKNOWN"
                    print(f"[ERROR SpecMix SpectrogramOut] {fname} with sample index {idx2_str}: {e}")
                    # traceback.print_exc()
                    # Fallback: revert to original audio if any SpecMix step failed
                    imitation_data = original_imitation_audio
                    reference_data = original_reference_audio
                    imitation_is_audio = True
                    reference_is_audio = True

        # --- Final Shape and Type Ensurance ---
        sample['imitation'] = imitation_data.cpu()
        sample['reference'] = reference_data.cpu()
        # sample['imitation'] = imitation_audio.squeeze().cpu()  # Ensure 1D tensor on CPU
        # sample['reference'] = reference_audio.squeeze().cpu()  # Ensure 1D tensor on CPU

        # Verify final shapes if debugging
        # print(f"[DEBUG End __getitem__] File: {fname}, Final shapes: im={sample['imitation'].shape}, ref={sample['reference'].shape}")

        if sample['imitation'].ndim != 1 or sample['reference'].ndim != 1:
            print(
                f"[FATAL SHAPE ERROR] File: {fname},Shapes NOT 1D! im:{sample['imitation'].shape}, ref:{sample['reference'].shape}")

        return sample