# audio_spec_mixer_from_spectrogram.py
import torch
import torchaudio
import torchaudio.transforms as T
import random
import numpy as np
import traceback


class SpecMixerSpectrogramOut:
    def __init__(self, sample_rate: int, n_fft: int = 1024, hop_length: int = 512,
                 gamma_range: tuple = (0.1, 0.5), max_time_bands: int = 3, max_freq_bands: int = 3,
                 mix_target_device: str = 'cpu',
                 n_mels: int = 128,
                 **kwargs):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.gamma_range = gamma_range
        self.max_time_bands = max_time_bands
        self.max_freq_bands = max_freq_bands
        self.device = torch.device(mix_target_device)

        self.spectrogram_transform = T.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            power=2.0,
            center=True
        ).to(self.device)

    def _to_tensor_and_device(self, waveform: np.ndarray) -> torch.Tensor:
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform.astype(np.float32))
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        return waveform.to(self.device)

    def _waveform_to_spectrogram(self, waveform_tensor: torch.Tensor) -> torch.Tensor:
        if waveform_tensor.ndim == 1:  # Ensure channel dimension
            waveform_tensor = waveform_tensor.unsqueeze(0)
        return self.spectrogram_transform(waveform_tensor)

    def _get_band_mask_details(self, spec_dim_size: int, min_relative_size: float, max_relative_size: float) -> tuple[
        int, int]:
        if spec_dim_size == 0:
            return 0, 0

        min_abs_size = max(1, int(min_relative_size * spec_dim_size))
        max_abs_size = max(1, int(max_relative_size * spec_dim_size))

        if min_abs_size > max_abs_size:
            min_abs_size = max_abs_size  # Ensure min <= max
        # Cap by dimension size (already implicitly handled by random.randint if max_abs_size is capped)
        max_abs_size = min(max_abs_size, spec_dim_size)
        min_abs_size = min(min_abs_size, max_abs_size)  # Ensure min_abs_size is also capped and not > max_abs_size

        band_sz = random.randint(min_abs_size, max_abs_size)  # Corrected from actual_max_bs

        start_max_range = spec_dim_size - band_sz
        band_start = random.randint(0, max(0, start_max_range))  # Ensure range is not negative
        band_end = band_start + band_sz
        return band_start, band_end

    def _generate_mix_mask(self, spec_shape: tuple) -> torch.Tensor:
        # spec_shape: (Channel, Freq, Time)
        _, n_freq_bins, n_time_frames = spec_shape

        mask = torch.zeros(n_freq_bins, n_time_frames, device=self.device)  # 0s for spec1, 1s for spec2
        min_rel_bs, max_rel_bs = self.gamma_range[0], self.gamma_range[1]

        num_freq_bands_to_apply = random.randint(0, self.max_freq_bands)
        for _ in range(num_freq_bands_to_apply):
            if n_freq_bins > 0:
                f_start, f_end = self._get_band_mask_details(n_freq_bins, min_rel_bs, max_rel_bs)
                mask[f_start:f_end, :] = 1.0

        num_time_bands_to_apply = random.randint(0, self.max_time_bands)
        for _ in range(num_time_bands_to_apply):
            if n_time_frames > 0:
                t_start, t_end = self._get_band_mask_details(n_time_frames, min_rel_bs, max_rel_bs)
                mask[:, t_start:t_end] = 1.0

        return mask.unsqueeze(0)  # Add channel dim: (1, Freq, Time)

    def apply_item_level_specmix(self, sample1_data_dict: dict, sample2_raw_audio_dict: dict,
                                 target_key: str = 'reference') -> dict:
        # Get audio waveform for sample1 from the input dict
        waveform1_audio_tensor = sample1_data_dict[target_key]
        if not isinstance(waveform1_audio_tensor, torch.Tensor):  # Ensure it's a tensor
            waveform1_audio_tensor = torch.from_numpy(np.array(waveform1_audio_tensor).astype(np.float32))
        # Ensure it's on the correct device and has a channel dimension if 1D
        waveform1_audio_tensor = self._to_tensor_and_device(waveform1_audio_tensor.cpu())

        # Get audio waveform for sample2 from the raw audio dict (which contains numpy arrays)
        waveform2_audio_np = sample2_raw_audio_dict[target_key]
        waveform2_audio_tensor = self._to_tensor_and_device(waveform2_audio_np)

        len1, len2 = waveform1_audio_tensor.shape[-1], waveform2_audio_tensor.shape[-1]
        num_freq_bins = self.n_fft // 2 + 1

        if len1 == 0 or len2 == 0:
            sample1_data_dict[target_key] = torch.zeros((num_freq_bins, 0), device=self.device).cpu()
            return sample1_data_dict

        if len1 != len2:
            min_len = min(len1, len2)
            waveform1_audio_tensor = waveform1_audio_tensor[..., :min_len]
            waveform2_audio_tensor = waveform2_audio_tensor[..., :min_len]

        output_spectrogram = None
        try:
            spec1_power = self._waveform_to_spectrogram(waveform1_audio_tensor)
            spec2_power = self._waveform_to_spectrogram(waveform2_audio_tensor)

            if spec1_power.shape[-1] == 0 or spec2_power.shape[-1] == 0:
                output_spectrogram = spec1_power  # Return original (possibly empty) spec
            elif spec1_power.shape != spec2_power.shape:
                output_spectrogram = spec1_power  # Return original if shapes mismatch
            else:
                mix_mask = self._generate_mix_mask(spec1_power.shape)
                mixed_spec_power = spec1_power * (1.0 - mix_mask) + spec2_power * mix_mask
                output_spectrogram = mixed_spec_power

        except Exception as e:
            # print(f"[ERROR SpecMixerSpectrogramOut] Error during spectrogram mixing for '{target_key}': {e}")
            # print(traceback.format_exc())
            # Fallback: convert original audio for sample1 to spectrogram
            try:
                output_spectrogram = self._waveform_to_spectrogram(waveform1_audio_tensor)
            except Exception:  # Final fallback to empty spec
                output_spectrogram = torch.zeros((1, num_freq_bins, 0), device=self.device)

        sample1_data_dict[target_key] = output_spectrogram.squeeze(0).cpu()  # Ensure (F,T) on CPU
        return sample1_data_dict