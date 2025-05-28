# audio_spec_mixer.py
import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
import random
import numpy as np
import traceback

class SpecMixer:
    def __init__(self, sample_rate: int, n_fft: int = 1024, hop_length: int = 512,
                 n_mels: int = 128, gamma_range: tuple = (0.1, 0.5), max_time_bands: int = 3, max_freq_bands: int = 3,
                 mix_target_device: str = 'cpu'):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.gamma_range = gamma_range
        self.max_time_bands = max_time_bands
        self.max_freq_bands = max_freq_bands
        self.device = torch.device(mix_target_device)

        self.mel_spectrogram_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=2.0,
            center=True
        ).to(self.device)

        self.inverse_mel_scale_transform = T.InverseMelScale(
            n_stft=self.n_fft // 2 + 1,
            n_mels=self.n_mels,
            sample_rate=self.sample_rate,
        ).to(self.device)

        self.griffin_lim_transform = T.GriffinLim(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            power=2.0,
            # --- MODIFICATION: Increased iterations ---
            n_iter=64  # Try 64 or even 100, up from 32
        ).to(self.device)

    def _to_tensor_and_device(self, waveform: np.ndarray) -> torch.Tensor:
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform.astype(np.float32))
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        return waveform.to(self.device)

    def _waveform_to_mel_spectrogram(self, waveform_tensor: torch.Tensor) -> torch.Tensor:
        if waveform_tensor.ndim == 1:
            waveform_tensor = waveform_tensor.unsqueeze(0)
        mel_spec_power = self.mel_spectrogram_transform(waveform_tensor)
        mel_spec_db = T.AmplitudeToDB(stype='power', top_db=80)(mel_spec_power)
        return mel_spec_db

    def _mel_spectrogram_to_waveform(self, mel_spectrogram_db: torch.Tensor,
                                     original_waveform_for_fallback: torch.Tensor) -> torch.Tensor:
        try:
            mel_spectrogram_power = F.DB_to_amplitude(mel_spectrogram_db, ref=1.0, power=0.5)

            # --- MODIFIED/SIMPLIFIED Regularization for mel_spectrogram_power ---
            # 1. Handle NaN/Inf
            if torch.isnan(mel_spectrogram_power).any() or torch.isinf(mel_spectrogram_power).any():
                mel_spectrogram_power = torch.nan_to_num(mel_spectrogram_power, nan=0.0, posinf=0.0, neginf=0.0)
            # 2. Ensure non-negativity and add a simple, small epsilon floor
            # This replaces your adaptive floor and structured noise for mel_spectrogram_power
            mel_spectrogram_power = torch.clamp(mel_spectrogram_power, min=0.0) + 1e-7


            linear_spec_approx = self.inverse_mel_scale_transform(mel_spectrogram_power)

            # --- MODIFIED/SIMPLIFIED Regularization for linear_spec_approx ---
            # 1. Handle NaN/Inf
            if torch.isnan(linear_spec_approx).any() or torch.isinf(linear_spec_approx).any():
                linear_spec_approx = torch.nan_to_num(linear_spec_approx, nan=0.0, posinf=0.0, neginf=0.0)
            # 2. Ensure non-negativity and add a simple, small epsilon floor
            # This replaces your adaptive floor and complex combined_reg for linear_spec_approx
            linear_spec_approx = torch.clamp(linear_spec_approx, min=0.0) + 1e-7


            waveform = self.griffin_lim_transform(linear_spec_approx)

            if torch.isnan(waveform).any() or torch.isinf(waveform).any():
                # This check is good. If GLI itself produces NaN/Inf, it's a failure.
                raise RuntimeError("NaN/Inf detected in Griffin-Lim output")

        except (RuntimeError, ValueError, Exception) as e:
            error_message_short = str(e)[:100]  # Keep a short version for the warning
            print(f"[WARNING SpecMixer] Reconstruction failed ({error_message_short}...), using original waveform")

            full_error_traceback = traceback.format_exc()
            print(f"[DEBUG SpecMixer Full Error Traceback]\n{full_error_traceback}")
            waveform = original_waveform_for_fallback.clone()

        if waveform.ndim > 1 and waveform.shape[0] == 1:
            waveform = waveform.squeeze(0)
        return waveform

    # _get_band, _generate_specmix_mask, apply_item_level_specmix remain the same as your uploaded file
    def _get_band(self, current_mask_shape_tuple: tuple, min_abs_size: int, max_abs_size: int, band_type: str,
                  mask_tensor: torch.Tensor) -> torch.Tensor:
        n_mels, time_frames = current_mask_shape_tuple

        if band_type.lower() == 'freq':
            dim_size = n_mels
        elif band_type.lower() == 'time':
            dim_size = time_frames
        else:
            raise ValueError("band_type must be 'freq' or 'time'")

        if dim_size == 0:  # Avoid issues if a dimension is zero
            return mask_tensor

        actual_min_bs = max(1, min(min_abs_size, dim_size))
        actual_max_bs = max(1, min(max_abs_size, dim_size))

        if actual_min_bs > actual_max_bs:
            actual_min_bs = actual_max_bs

        band_sz = random.randint(actual_min_bs, actual_max_bs)

        start_max_range = dim_size - band_sz
        band_start = random.randint(0, max(0, start_max_range))
        band_end = band_start + band_sz

        if band_type.lower() == 'freq':
            mask_tensor[band_start:band_end, :] = 0.0
        elif band_type.lower() == 'time':
            mask_tensor[:, band_start:band_end] = 0.0
        return mask_tensor

    def _generate_specmix_mask(self, spec_shape: tuple) -> torch.Tensor:
        _, n_mels, time_frames = spec_shape
        mask = torch.ones(n_mels, time_frames, device=self.device)

        min_freq_band_abs = int(self.gamma_range[0] * n_mels)
        max_freq_band_abs = int(self.gamma_range[1] * n_mels)
        min_time_band_abs = int(self.gamma_range[0] * time_frames)
        max_time_band_abs = int(self.gamma_range[1] * time_frames)

        num_freq_bands_to_apply = random.randint(0, self.max_freq_bands)
        for _ in range(num_freq_bands_to_apply):
            if n_mels > 0:
                mask = self._get_band((n_mels, time_frames),
                                      min_freq_band_abs, max_freq_band_abs,
                                      'freq', mask)

        num_time_bands_to_apply = random.randint(0, self.max_time_bands)
        for _ in range(num_time_bands_to_apply):
            if time_frames > 0:
                mask = self._get_band((n_mels, time_frames),
                                      min_time_band_abs, max_time_band_abs,
                                      'time', mask)

        return mask.unsqueeze(0)

    def apply_item_level_specmix(self, sample1: dict, sample2: dict,
                                 target_key: str = 'reference') -> dict:

        waveform1_np = sample1[target_key].numpy() if isinstance(sample1[target_key], torch.Tensor) else np.array(
            sample1[target_key])
        waveform2_np = sample2[target_key].numpy() if isinstance(sample2[target_key], torch.Tensor) else np.array(
            sample2[target_key])

        waveform1_tensor = self._to_tensor_and_device(waveform1_np)
        waveform2_tensor = self._to_tensor_and_device(waveform2_np)

        len1, len2 = waveform1_tensor.shape[-1], waveform2_tensor.shape[-1]
        if len1 == 0 or len2 == 0:
            # print(f"[WARNING SpecMixer] Empty waveform detected for {target_key}. Skipping mix.")
            return sample1

        if len1 != len2:
            min_len = min(len1, len2)
            waveform1_tensor = waveform1_tensor[..., :min_len]
            waveform2_tensor = waveform2_tensor[..., :min_len]

        spec1 = self._waveform_to_mel_spectrogram(waveform1_tensor)
        spec2 = self._waveform_to_mel_spectrogram(waveform2_tensor)

        if spec1.shape != spec2.shape:
            # print(f"[WARNING SpecMixer] Spectrogram shapes mismatch. Skipping mix for '{target_key}'.")
            return sample1

        mask = self._generate_specmix_mask(spec1.shape)
        mixed_spec = mask * spec1 + (1.0 - mask) * spec2

        mixed_waveform_tensor = self._mel_spectrogram_to_waveform(mixed_spec, waveform1_tensor)

        sample1[target_key] = mixed_waveform_tensor.squeeze().cpu()
        # print(f"[DEBUG SpecMixer] Applied SpecMix to '{target_key}'")
        return sample1