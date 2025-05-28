# aug_dataset_full_2.py
import re
import torch
import numpy as np
import random
import traceback  # For debugging SpecMix errors
from torch_audiomentations import Compose as TorchCompose, Gain
from audiomentations import Compose as NumpyCompose, PitchShift as NumpyPitchShift, TimeStretch as NumpyTimeStretch, \
    AddGaussianNoise as NumpyAddGaussianNoise
from qvim_mn_baseline.dataset import VimSketchDataset
from audio_temporal_modifier import TemporalModifier
from audio_spec_mixer_from_spectrogram import SpecMixerSpectrogramOut  # Use the spectrogram-out version


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
            specmix_n_mels: int = 128,  # Will be passed but ignored by SpecMixerSpectrogramOut
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
                self.spec_mixer = SpecMixerSpectrogramOut(  # Using the spectrogram-out version
                    sample_rate=self.sample_rate,
                    n_fft=specmix_n_fft,
                    hop_length=specmix_hop_length,
                    n_mels=specmix_n_mels,  # Passed for compatibility, ignored by mixer
                    gamma_range=specmix_gamma_range,
                    max_time_bands=specmix_max_bands,
                    max_freq_bands=specmix_max_bands,
                    mix_target_device=specmix_device
                )
            else:
                self.spec_mixer = None
        else:  # Not augmenting
            self.numpy_audio_pipeline = None
            self.torch_audio_pipeline = None
            self.temporal_modifier = None
            self.spec_mixer = None

    def __getitem__(self, index):
        sample = super().__getitem__(index)  # Contains numpy audio arrays
        fname = sample['reference_filename']

        # Convert to PyTorch tensors; these variables will hold the current state of the data
        imitation_payload = torch.from_numpy(sample['imitation'].astype(np.float32))
        reference_payload = torch.from_numpy(sample['reference'].astype(np.float32))

        if self.augment:
            # --- Initial Audio Augmentations (Numpy, TorchAudio, Temporal) ---
            # These operate on and update 'imitation_payload' and 'reference_payload'
            # which are currently audio tensors.

            current_imitation_for_audio_aug = imitation_payload.clone()
            current_reference_for_audio_aug = reference_payload.clone()
            audio_augmented_this_sample = False  # Flag if any audio aug was applied

            # (Your existing class-targeted and general augmentation probability logic)
            # Simplified for brevity, assuming 'apply_this_round_of_audio_augs' is determined
            apply_this_round_of_audio_augs = False
            # Re-insert your MRR-based probability logic here to set apply_this_round_of_audio_augs = True if augs should run
            fname_lower = fname.lower()
            best_matching_class_info = None
            if self.class_info:
                max_token_matches = 0
                for class_name_iter, info_iter in self.class_info.items():
                    count = sum(tok in fname_lower for tok in info_iter['tokens'])
                    if count > max_token_matches:
                        max_token_matches = count
                        best_matching_class_info = info_iter

            aug_decision_prob = self.base_augmentation_prob
            if best_matching_class_info:
                mrr_score = best_matching_class_info['mrr']
                aug_decision_prob = self.base_augmentation_prob + (1.0 - mrr_score) * self.mrr_influence_factor
                aug_decision_prob = max(0.0, min(1.0, aug_decision_prob))

            if random.random() < aug_decision_prob:
                apply_this_round_of_audio_augs = True

            if apply_this_round_of_audio_augs:
                audio_augmented_this_sample = True
                if self.numpy_audio_pipeline:
                    try:
                        im_np = current_imitation_for_audio_aug.cpu().numpy()
                        ref_np = current_reference_for_audio_aug.cpu().numpy()
                        im_aug_np = self.numpy_audio_pipeline(samples=im_np, sample_rate=self.sample_rate)
                        ref_aug_np = self.numpy_audio_pipeline(samples=ref_np, sample_rate=self.sample_rate)
                        current_imitation_for_audio_aug = torch.from_numpy(im_aug_np.astype(np.float32))
                        current_reference_for_audio_aug = torch.from_numpy(ref_aug_np.astype(np.float32))
                        # print(f"[DEBUG NumpyAug Applied] File: {fname}")
                    except Exception as e:
                        print(f"[ERROR NumpyAug] {fname}: {e}")

                if self.torch_audio_pipeline:
                    try:
                        im_in = current_imitation_for_audio_aug.unsqueeze(0).unsqueeze(0).to(self.target_device)
                        ref_in = current_reference_for_audio_aug.unsqueeze(0).unsqueeze(0).to(self.target_device)
                        current_imitation_for_audio_aug = \
                        self.torch_audio_pipeline(samples=im_in, sample_rate=self.sample_rate)[
                            'samples'].squeeze().cpu()
                        current_reference_for_audio_aug = \
                        self.torch_audio_pipeline(samples=ref_in, sample_rate=self.sample_rate)[
                            'samples'].squeeze().cpu()
                        # print(f"[DEBUG TorchAug Applied] File: {fname}")
                    except Exception as e:
                        print(f"[ERROR TorchAug] {fname}: {e}")

                if self.temporal_modifier and random.random() < self.apply_envelope_modulation_prob:
                    temp_sample_for_tm = {'imitation': current_imitation_for_audio_aug.cpu().numpy(),
                                          'reference': current_reference_for_audio_aug.cpu().numpy()}
                    try:
                        mod_tm = self.temporal_modifier.apply_envelope_modulation(temp_sample_for_tm)
                        current_imitation_for_audio_aug = torch.as_tensor(mod_tm['imitation'], dtype=torch.float32)
                        current_reference_for_audio_aug = torch.as_tensor(mod_tm['reference'], dtype=torch.float32)
                        # print(f"[DEBUG TemporalMod Applied] File: {fname}")
                    except Exception as e:
                        print(f"[ERROR TemporalMod] {fname}: {e}")

            if audio_augmented_this_sample:
                imitation_payload = current_imitation_for_audio_aug
                reference_payload = current_reference_for_audio_aug

            # --- SpecMix (operates on audio, outputs spectrograms) ---
            if self.spec_mixer and random.random() < self.apply_specmix_prob:
                # Store current audio state for fallback if SpecMix fails internally
                audio_imitation_before_specmix = imitation_payload.clone()
                audio_reference_before_specmix = reference_payload.clone()
                idx2 = -1  # Initialize idx2 for safe error printing
                try:
                    idx2 = random.randint(0, len(self) - 1)
                    while idx2 == index: idx2 = random.randint(0, len(self) - 1)
                    sample2_raw_audio_dict = super().__getitem__(idx2)  # dict of numpy arrays for sample2

                    # Process 'reference'
                    # Pass a dict containing the audio for the target key
                    sample1_for_ref_processing = {'reference': audio_reference_before_specmix}
                    processed_ref_dict = self.spec_mixer.apply_item_level_specmix(
                        sample1_for_ref_processing,
                        sample2_raw_audio_dict,
                        target_key='reference'
                    )
                    reference_payload = processed_ref_dict['reference']  # Now a spectrogram (F,T)

                    # Process 'imitation'
                    sample1_for_im_processing = {'imitation': audio_imitation_before_specmix}
                    processed_im_dict = self.spec_mixer.apply_item_level_specmix(
                        sample1_for_im_processing,
                        sample2_raw_audio_dict,
                        target_key='imitation'
                    )
                    imitation_payload = processed_im_dict['imitation']  # Now a spectrogram (F,T)

                    # print(f"[DEBUG SpecMix Applied (Outputting Spectrogram)] File: {fname}, Shapes: im={imitation_payload.shape}, ref={reference_payload.shape}")

                except Exception as e:
                    idx2_str = str(idx2) if idx2 != -1 else "UNKNOWN (error before idx2 assignment or in SpecMix)"
                    print(f"[ERROR SpecMix SpectrogramOut Procedure] File: {fname}, Sample2 Idx: {idx2_str}: {e}")
                    # print(traceback.format_exc()) # Uncomment for full traceback
                    # Fallback to audio state before SpecMix attempt
                    imitation_payload = audio_imitation_before_specmix
                    reference_payload = audio_reference_before_specmix

        # --- Final Data Consistency Check and Assignment ---
        # If self.spec_mixer is active (meaning spectrogram output was intended for at least some items
        # if the probability hit), ensure ALL items returned by __getitem__ for this dataset instance
        # become spectrograms for batching consistency in the DataLoader.
        if self.spec_mixer:  # True if apply_specmix_prob > 0 in __init__
            # Check and convert imitation_payload if it's still audio
            if imitation_payload.ndim == 1:  # Heuristic: 1D tensor is audio
                # print(f"[DEBUG __getitem__ consistency] Converting IMITATION audio to spec for: {fname if self.augment else 'N/A'}")
                try:
                    # Ensure it's on the device the mixer's transform expects, then convert
                    # The spec_mixer's methods expect audio on its device.
                    # imitation_payload is already a tensor here.
                    audio_tensor_for_conversion = self.spec_mixer._to_tensor_and_device(imitation_payload.cpu())
                    imitation_payload = self.spec_mixer._waveform_to_spectrogram(
                        audio_tensor_for_conversion).squeeze(0).cpu()
                except Exception as conversion_e:
                    # Fallback to an empty spectrogram or handle error appropriately
                    # This error would likely only happen if imitation_payload was not valid audio
                    # or if self.spec_mixer somehow failed its own conversion (unlikely for _waveform_to_spectrogram)
                    print(
                        f"[ERROR __getitem__ consistency] Failed to convert IMITATION audio to spec for {fname if self.augment else 'N/A'}: {conversion_e}")
                    num_freq_bins = self.spec_mixer.n_fft // 2 + 1
                    imitation_payload = torch.zeros((num_freq_bins, 0), device='cpu')  # Empty spec (F, T)

            # Check and convert reference_payload if it's still audio
            if reference_payload.ndim == 1:  # Heuristic: 1D tensor is audio
                # print(f"[DEBUG __getitem__ consistency] Converting REFERENCE audio to spec for: {fname if self.augment else 'N/A'}")
                try:
                    audio_tensor_for_conversion = self.spec_mixer._to_tensor_and_device(reference_payload.cpu())
                    reference_payload = self.spec_mixer._waveform_to_spectrogram(
                        audio_tensor_for_conversion).squeeze(0).cpu()
                except Exception as conversion_e:
                    print(
                        f"[ERROR __getitem__ consistency] Failed to convert REFERENCE audio to spec for {fname if self.augment else 'N/A'}: {conversion_e}")
                    num_freq_bins = self.spec_mixer.n_fft // 2 + 1
                    reference_payload = torch.zeros((num_freq_bins, 0), device='cpu')  # Empty spec (F, T)

        # Now, if self.spec_mixer was active (configured in __init__),
        # imitation_payload and reference_payload should BOTH be spectrograms (Freq, Time).
        # If self.spec_mixer was None (apply_specmix_prob was 0), they remain audio tensors (L,).
        # This consistency (all audio OR all spectrograms for a given dataset instance)
        # allows the default collate_fn to work.

        sample['imitation'] = imitation_payload.cpu()  # Ensure CPU for dataloader
        sample['reference'] = reference_payload.cpu()  # Ensure CPU for dataloader

        # The FATAL SHAPE ERROR check for ndim != 1 was removed from your code,
        # which is correct as the payload can now be a 2D spectrogram.
        # AugmentMelSTFT is responsible for handling both audio and spectrogram inputs.

        return sample