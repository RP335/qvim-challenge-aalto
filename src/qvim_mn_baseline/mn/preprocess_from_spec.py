# preprocess.py
import torch.nn as nn
import torchaudio
import torch
import contextlib

class AugmentMelSTFT(nn.Module):
    def __init__(self, n_mels=128, sr=32000, win_length=800, hopsize=320, n_fft=1024, freqm=48, timem=192,
                 fmin=0.0, fmax=None, fmin_aug_range=10, fmax_aug_range=2000):
        torch.nn.Module.__init__(self)
        self.win_length = win_length
        self.n_mels = n_mels
        self.n_fft = n_fft  # n_fft is crucial for get_mel_banks
        self.sr = sr
        self.fmin = fmin
        if fmax is None:
            fmax_default = sr // 2 - fmax_aug_range // 2
            # print(f"Warning: FMAX is None setting to {fmax_default} in AugmentMelSTFT")
            self.fmax = fmax_default
        else:
            self.fmax = fmax
        self.hopsize = hopsize  # hopsize is used for STFT on audio
        self.register_buffer('window',
                             torch.hann_window(win_length, periodic=False),
                             persistent=False)
        assert fmin_aug_range >= 1, f"fmin_aug_range={fmin_aug_range} should be >=1; 1 means no augmentation"
        assert fmax_aug_range >= 1, f"fmax_aug_range={fmax_aug_range} should be >=1; 1 means no augmentation"
        self.fmin_aug_range = fmin_aug_range
        self.fmax_aug_range = fmax_aug_range

        self.register_buffer("preemphasis_coefficient", torch.as_tensor([[[-.97, 1]]]), persistent=False)
        if freqm == 0:
            self.freqm = torch.nn.Identity()
        else:
            self.freqm = torchaudio.transforms.FrequencyMasking(freqm, iid_masks=True)
        if timem == 0:
            self.timem = torch.nn.Identity()
        else:
            self.timem = torchaudio.transforms.TimeMasking(timem, iid_masks=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x can be an audio waveform [B, L] or [B, 1, L]
        # OR a pre-computed power spectrogram [B, Freq, Time] or [B, 1, Freq, Time]

        # Input may already be on target device from dataloader
        # x = x.to(self.window.device) # Ensure x is on the same device as buffers

        power_spectrogram: torch.Tensor

        # Check if input is likely an audio waveform or a pre-computed spectrogram
        # A simple heuristic: waveforms are 1D (time) or 2D (batch, time) / (channel, time)
        # Spectrograms are 2D (freq, time) or 3D (channel, freq, time) or 4D (batch, channel, freq, time)
        # Let's assume if ndim > 2, it's a batched spectrogram (B, C, F, T) or (B, F, T)
        # And if ndim == 2, it could be (B,L) audio or (F,T) spec. We'll use shape comparison.
        # If dataloader gives (B,L) for audio, and (B,F,T) or (B,1,F,T) for spec from SpecMixer.

        is_waveform = True  # Assume waveform by default
        if x.ndim == 3:  # Potentially (Batch, Freq, Time) or (Batch, Channel=1, Time_Waveform)
            # If last dim is much larger than second to last, likely waveform (B, 1, L_audio)
            # If second to last dim is n_fft // 2 + 1, likely spectrogram (B, F, T)
            if x.shape[1] == (self.n_fft // 2 + 1):  # Matches expected Freq bins
                is_waveform = False
            # Add more checks if needed, e.g. x.shape[1] > x.shape[2] for (B,L_waveform_channel_squeezed, T_short)
        elif x.ndim == 4:  # Definitely a batched spectrogram (B, C, F, T)
            is_waveform = False
        elif x.ndim == 2 and x.shape[0] != 1 and x.shape[0] == (self.n_fft // 2 + 1):  # (F,T) single spec case
            is_waveform = False

        if is_waveform:
            # print(f"[DEBUG AugmentMelSTFT] Input detected as waveform, shape: {x.shape}")
            # Ensure x has a channel dimension for conv1d if it's [B, L]
            if x.ndim == 2 and x.shape[0] > 1:  # Assuming [B, L]
                x_for_conv = x.unsqueeze(1)
            elif x.ndim == 1:  # Assuming [L]
                x_for_conv = x.unsqueeze(0).unsqueeze(0)  # [1,1,L]
            else:  # Assuming [B, 1, L] or [1,L]
                x_for_conv = x if x.ndim == 3 else x.unsqueeze(0)

            # Ensure x_for_conv is on the same device as preemphasis_coefficient
            x_for_conv = x_for_conv.to(self.preemphasis_coefficient.device)

            processed_x = nn.functional.conv1d(x_for_conv, self.preemphasis_coefficient).squeeze(1)

            # STFT expects [..., Time]
            # If processed_x became [B, Time], it's fine.
            # If it was [1, Time] from a single [L] input, it's also fine.
            complex_stft = torch.stft(processed_x, self.n_fft, hop_length=self.hopsize, win_length=self.win_length,
                                      center=True, normalized=False, window=self.window.to(processed_x.device),
                                      return_complex=True)
            # power_spectrogram = complex_stft.abs().pow(2.0) # More direct power calculation
            real_stft = torch.view_as_real(complex_stft)
            power_spectrogram = (real_stft ** 2).sum(dim=-1)  # (Batch, Freq, Time) or (Freq,Time)
        else:
            # print(f"[DEBUG AugmentMelSTFT] Input detected as spectrogram, shape: {x.shape}")
            # Input x is already a power spectrogram, potentially (B, C, F, T) or (B, F, T)
            # Ensure it has the channel dimension squeezed if it's (B, 1, F, T) for consistency before mel
            if x.ndim == 4 and x.shape[1] == 1:  # (B, 1, F, T)
                power_spectrogram = x.squeeze(1)  # (B, F, T)
            elif x.ndim == 3 and x.shape[0] != 1 and x.shape[1] == (self.n_fft // 2 + 1):  # (B, F, T) case
                power_spectrogram = x
            elif x.ndim == 2 and x.shape[0] == (self.n_fft // 2 + 1):  # (F,T) case, add batch dim
                power_spectrogram = x.unsqueeze(0)
            else:
                # Fallback or error if shape is unexpected for spectrogram
                # print(f"[ERROR AugmentMelSTFT] Unexpected spectrogram input shape: {x.shape}")
                # As a fallback, try to compute STFT assuming it was audio after all (might error)
                x_for_conv = x.unsqueeze(1) if x.ndim == 2 else x  # make it [B,1,L] or use as is if [B,C,L]
                x_for_conv = x_for_conv.to(self.preemphasis_coefficient.device)
                processed_x = nn.functional.conv1d(x_for_conv, self.preemphasis_coefficient).squeeze(1)
                complex_stft = torch.stft(processed_x, self.n_fft, hop_length=self.hopsize, win_length=self.win_length,
                                          center=True, normalized=False, window=self.window.to(processed_x.device),
                                          return_complex=True)
                real_stft = torch.view_as_real(complex_stft)
                power_spectrogram = (real_stft ** 2).sum(dim=-1)

        fmin_val = self.fmin
        fmax_val = self.fmax
        if self.training:  # Apply fmin/fmax augmentation only during training
            fmin_val = self.fmin + torch.randint(0, self.fmin_aug_range + 1, (1,)).item()
            fmax_val = self.fmax + self.fmax_aug_range // 2 - torch.randint(0, self.fmax_aug_range + 1, (1,)).item()
            # Ensure fmin < fmax
            if fmin_val >= fmax_val: fmin_val = self.fmin  # revert to default if aug makes it invalid
            fmax_val = max(fmin_val + self.n_mels, fmax_val)  # ensure fmax can support n_mels

        mel_basis, _ = torchaudio.compliance.kaldi.get_mel_banks(self.n_mels, self.n_fft, self.sr,
                                                                 fmin_val, fmax_val,
                                                                 vtln_low=100.0, vtln_high=-500., vtln_warp_factor=1.0)
        mel_basis = torch.as_tensor(torch.nn.functional.pad(mel_basis, (0, 1), mode='constant', value=0),
                                    device=power_spectrogram.device)  # Ensure mel_basis is on same device

        # Ensure power_spectrogram is (Batch, Freq, Time) for matmul
        # or handle unbatched (Freq,Time) -> (1,Freq,Time) if necessary before this
        if power_spectrogram.ndim == 2:  # (F,T)
            power_spectrogram_for_matmul = power_spectrogram.unsqueeze(0)
        else:  # Assumed (B,F,T)
            power_spectrogram_for_matmul = power_spectrogram

        # Perform matrix multiplication for mel spectrogram
        # mel_basis is (n_mels, n_freq_bins), power_spectrogram is (B, n_freq_bins, T)
        # Result should be (B, n_mels, T)
        with torch.cuda.amp.autocast(enabled=False) if torch.cuda.is_available() else contextlib.nullcontext():
            melspec = torch.matmul(mel_basis.to(power_spectrogram_for_matmul.dtype),
                                   power_spectrogram_for_matmul.to(mel_basis.dtype))

        melspec = (melspec + 0.00001).log()  # Log scale

        if self.training:  # Apply SpecAugment only during training
            melspec = self.freqm(melspec)
            melspec = self.timem(melspec)

        melspec = (melspec + 4.5) / 5.  # fast normalization

        return melspec