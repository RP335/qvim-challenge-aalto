# audio_spec_mixer.py
import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
import random
import numpy as np


class SpecMixer:
    def __init__(self, sample_rate: int, n_fft: int = 1024, hop_length: int = 512,
                 n_mels: int = 128, gamma_range: tuple = (0.1, 0.5), max_time_bands: int = 3, max_freq_bands: int = 3,
                 mix_target_device: str = 'cpu'):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.gamma_range = gamma_range  # Range for relative band sizes
        self.max_time_bands = max_time_bands  # Max number of time bands to apply
        self.max_freq_bands = max_freq_bands  # Max number of frequency bands to apply
        self.device = torch.device(mix_target_device)

        self.mel_spectrogram_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=2.0,  # Outputs power spectrogram
            center=True
        ).to(self.device)

        self.inverse_mel_scale_transform = T.InverseMelScale(
            n_stft=self.n_fft // 2 + 1,
            n_mels=self.n_mels,
            sample_rate=self.sample_rate,
            # norm="slaney", # common normalization
            # mel_scale="slaney" # common scale
        ).to(self.device)

        self.griffin_lim_transform = T.GriffinLim(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            power=2.0,  # Expects power spectrogram
            n_iter=32  # Default is 32, can be adjusted
        ).to(self.device)

    def _to_tensor_and_device(self, waveform: np.ndarray) -> torch.Tensor:
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform.astype(np.float32))
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)  # Add channel dimension
        return waveform.to(self.device)

    def _waveform_to_mel_spectrogram(self, waveform_tensor: torch.Tensor) -> torch.Tensor:
        if waveform_tensor.ndim == 1:
            waveform_tensor = waveform_tensor.unsqueeze(0)
        mel_spec_power = self.mel_spectrogram_transform(waveform_tensor)
        # Convert power spectrogram to dB scale for mixing
        mel_spec_db = T.AmplitudeToDB(stype='power', top_db=80)(mel_spec_power)
        return mel_spec_db

    def _mel_spectrogram_to_waveform(self, mel_spectrogram_db: torch.Tensor,
                                     original_waveform_for_fallback: torch.Tensor) -> torch.Tensor:
        try:
            # Convert dB-scaled mel spectrogram back to power mel spectrogram
            mel_spectrogram_power = F.DB_to_amplitude(mel_spectrogram_db, ref=1.0, power=0.5)

            # Enhanced robustness: Clean and validate mel_spectrogram_power
            if torch.isnan(mel_spectrogram_power).any() or torch.isinf(mel_spectrogram_power).any():
                mel_spectrogram_power = torch.nan_to_num(mel_spectrogram_power, nan=0.0, posinf=0.0, neginf=0.0)

            mel_spectrogram_power = torch.clamp(mel_spectrogram_power, min=1e-8)  # Larger epsilon

            # Add regularization to prevent rank deficiency
            noise_level = 1e-6 * torch.mean(mel_spectrogram_power)
            mel_spectrogram_power = mel_spectrogram_power + noise_level * torch.rand_like(mel_spectrogram_power)

            # Convert power mel spectrogram to linear power spectrogram
            linear_spec_approx = self.inverse_mel_scale_transform(mel_spectrogram_power)

            # Enhanced robustness: Clean and validate linear_spec_approx
            if torch.isnan(linear_spec_approx).any() or torch.isinf(linear_spec_approx).any():
                linear_spec_approx = torch.nan_to_num(linear_spec_approx, nan=0.0, posinf=0.0, neginf=0.0)

            linear_spec_approx = torch.clamp(linear_spec_approx, min=1e-8)

            # Add regularization to linear spectrogram as well
            noise_level = 1e-6 * torch.mean(linear_spec_approx)
            linear_spec_approx = linear_spec_approx + noise_level * torch.rand_like(linear_spec_approx)

            # Ensure minimum rank by adding small diagonal regularization
            if linear_spec_approx.shape[-2] > 1:  # Only if we have multiple frequency bins
                regularization = torch.eye(linear_spec_approx.shape[-2], device=self.device) * 1e-7
                # Broadcast regularization across time frames
                reg_expanded = regularization.unsqueeze(-1).expand(-1, -1, linear_spec_approx.shape[-1])
                if linear_spec_approx.ndim == 3:  # (batch, freq, time)
                    linear_spec_approx = linear_spec_approx + reg_expanded.unsqueeze(0)
                else:  # (freq, time)
                    linear_spec_approx = linear_spec_approx + reg_expanded.squeeze(0)

            # Convert linear power spectrogram to waveform using Griffin-Lim
            waveform = self.griffin_lim_transform(linear_spec_approx)

            # Final validation
            if torch.isnan(waveform).any() or torch.isinf(waveform).any():
                raise RuntimeError("NaN/Inf detected in Griffin-Lim output")

        except (RuntimeError, ValueError, Exception) as e:
            # Enhanced fallback: return original waveform instead of silence
            print(f"[WARNING SpecMixer] Reconstruction failed ({str(e)[:50]}...), using original waveform")
            waveform = original_waveform_for_fallback.clone()

        if waveform.ndim > 1 and waveform.shape[0] == 1:  # Ensure output is (num_samples) if mono
            waveform = waveform.squeeze(0)
        return waveform

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

        # Pass the original waveform1_tensor (which has the correct length before potential STFT frame changes)
        # for fallback calculation if Griffin-Lim fails.
        mixed_waveform_tensor = self._mel_spectrogram_to_waveform(mixed_spec, waveform1_tensor)

        sample1[target_key] = mixed_waveform_tensor.squeeze().cpu()
        # print(f"[DEBUG SpecMixer] Applied SpecMix to '{target_key}'")
        return sample1