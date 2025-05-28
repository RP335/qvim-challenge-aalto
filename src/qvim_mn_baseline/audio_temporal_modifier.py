import numpy as np
import librosa  # For RMS calculation


class TemporalModifier:
    def __init__(self, sample_rate: int, rms_frame_length: int = 2048, rms_hop_length: int = 512):
        self.sample_rate = sample_rate
        self.rms_frame_length = rms_frame_length
        self.rms_hop_length = rms_hop_length

    def _calculate_rms_envelope(self, audio_np: np.ndarray) -> np.ndarray:
        # Calculate RMS envelope
        rms_envelope = librosa.feature.rms(y=audio_np,
                                           frame_length=self.rms_frame_length,
                                           hop_length=self.rms_hop_length)[0]
        # Normalize envelope (e.g., to [0, 1] or simply by its max)
        if np.max(rms_envelope) > 1e-6:  # Avoid division by zero
            rms_envelope = rms_envelope / np.max(rms_envelope)
        else:
            rms_envelope = np.zeros_like(rms_envelope)  # or ones_like if preferred for silence
        return rms_envelope

    def _align_envelope_to_audio(self, envelope: np.ndarray, audio_length: int) -> np.ndarray:
        # Upsample/interpolate envelope to match audio length
        # This can be simple repeating or linear interpolation
        times_envelope = librosa.frames_to_time(np.arange(envelope.shape[0]),
                                                sr=self.sample_rate,
                                                hop_length=self.rms_hop_length)
        times_audio = librosa.times_like(np.empty(audio_length),
                                         sr=self.sample_rate)  # Creates time array for audio length

        aligned_envelope = np.interp(times_audio, times_envelope, envelope)
        return aligned_envelope

    def apply_envelope_modulation(self, sample: dict,
                                  modulate_reference_with_imitation: bool = True,
                                  gain_factor: float = 1.0) -> dict:
        """
        Modulates one audio in the sample pair using the envelope of the other.
        The input sample dict should contain 'imitation' and 'reference' as numpy arrays or tensors.
        It returns the modified sample dict with numpy arrays converted back to tensors if they were initially.
        """

        imitation_np = sample['imitation'].numpy() if isinstance(sample['imitation'], torch.Tensor) else np.array(
            sample['imitation'])
        reference_np = sample['reference'].numpy() if isinstance(sample['reference'], torch.Tensor) else np.array(
            sample['reference'])

        if modulate_reference_with_imitation:
            source_for_envelope = imitation_np
            target_to_modulate = reference_np
        else:  # Modulate imitation with reference envelope (less common for QBE but possible)
            source_for_envelope = reference_np
            target_to_modulate = imitation_np

        if len(source_for_envelope) == 0 or len(target_to_modulate) == 0:
            return sample  # Nothing to modulate

        envelope = self._calculate_rms_envelope(source_for_envelope)
        aligned_envelope = self._align_envelope_to_audio(envelope, len(target_to_modulate))

        modulated_audio = target_to_modulate * (aligned_envelope * gain_factor)

        # Ensure clipping if necessary, though direct multiplication might not require it
        # modulated_audio = np.clip(modulated_audio, -1.0, 1.0) # If your audio is in [-1,1]

        if modulate_reference_with_imitation:
            sample['reference'] = torch.from_numpy(modulated_audio.astype(np.float32))
        else:
            sample['imitation'] = torch.from_numpy(modulated_audio.astype(np.float32))

        return sample