# Debug script to identify the audiomentations issue

import sys
import traceback
import numpy as np


# 1. First, let's check what we have installed
def check_installation():
    print("=== CHECKING INSTALLATIONS ===")
    try:
        import audiomentations
        print(f"audiomentations version: {audiomentations.__version__}")
        print(f"audiomentations path: {audiomentations.__file__}")
    except Exception as e:
        print(f"Error importing audiomentations: {e}")

    try:
        import torch_audiomentations
        print(f"torch_audiomentations found: {torch_audiomentations.__file__}")
        print("WARNING: You have torch_audiomentations installed!")
    except ImportError:
        print("torch_audiomentations not found (good)")

    try:
        import librosa
        print(f"librosa version: {librosa.__version__}")
    except Exception as e:
        print(f"librosa issue: {e}")


# 2. Test each transform individually
def test_individual_transforms():
    print("\n=== TESTING INDIVIDUAL TRANSFORMS ===")

    # Generate test audio
    test_audio = np.random.uniform(low=-0.2, high=0.2, size=(16000,)).astype(np.float32)
    sample_rate = 16000

    # Test AddGaussianNoise
    try:
        from audiomentations import AddGaussianNoise
        noise_transform = AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1.0)
        result = noise_transform(samples=test_audio, sample_rate=sample_rate)
        print("✓ AddGaussianNoise works")
    except Exception as e:
        print(f"✗ AddGaussianNoise failed: {e}")
        traceback.print_exc()

    # Test TimeStretch
    try:
        from audiomentations import TimeStretch
        stretch_transform = TimeStretch(min_rate=0.8, max_rate=1.2, p=1.0)
        result = stretch_transform(samples=test_audio, sample_rate=sample_rate)
        print("✓ TimeStretch works")
    except Exception as e:
        print(f"✗ TimeStretch failed: {e}")
        traceback.print_exc()

    # Test PitchShift - THIS IS THE PROBLEM CHILD
    try:
        from audiomentations import PitchShift
        print(f"PitchShift class: {PitchShift}")
        print(f"PitchShift module: {PitchShift.__module__}")

        pitch_transform = PitchShift(min_semitones=-2.0, max_semitones=2.0, p=1.0)
        print(f"Created PitchShift transform: {pitch_transform}")

        # Try to call it
        result = pitch_transform(samples=test_audio, sample_rate=sample_rate)
        print("✓ PitchShift works")
    except Exception as e:
        print(f"✗ PitchShift failed: {e}")
        traceback.print_exc()


# 3. Test the composition
def test_compose():
    print("\n=== TESTING COMPOSE ===")

    test_audio = np.random.uniform(low=-0.2, high=0.2, size=(16000,)).astype(np.float32)
    sample_rate = 16000

    try:
        from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift

        # Test without PitchShift first
        pipeline_no_pitch = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1.0),
            TimeStretch(min_rate=0.8, max_rate=1.2, p=1.0),
        ])
        result = pipeline_no_pitch(samples=test_audio, sample_rate=sample_rate)
        print("✓ Compose without PitchShift works")

        # Now test with PitchShift
        pipeline_with_pitch = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1.0),
            PitchShift(min_semitones=-2.0, max_semitones=2.0, p=1.0),
            TimeStretch(min_rate=0.8, max_rate=1.2, p=1.0),
        ])
        result = pipeline_with_pitch(samples=test_audio, sample_rate=sample_rate)
        print("✓ Compose with PitchShift works")

    except Exception as e:
        print(f"✗ Compose failed: {e}")
        traceback.print_exc()


# 4. Let's check what's actually being called
def inspect_pitch_shift():
    print("\n=== INSPECTING PITCHSHIFT ===")

    try:
        from audiomentations import PitchShift
        import inspect

        # Get the source code if possible
        try:
            source = inspect.getsource(PitchShift.__call__)
            print("PitchShift.__call__ method found")
            print("First 500 chars of source:")
            print(source[:500])
        except:
            print("Could not get source code")

        # Check the signature
        sig = inspect.signature(PitchShift.__call__)
        print(f"PitchShift.__call__ signature: {sig}")

    except Exception as e:
        print(f"Inspection failed: {e}")


# 5. Alternative: Test with different backend
def test_pitch_shift_backends():
    print("\n=== TESTING PITCHSHIFT BACKENDS ===")

    test_audio = np.random.uniform(low=-0.2, high=0.2, size=(16000,)).astype(np.float32)
    sample_rate = 16000

    try:
        from audiomentations import PitchShift

        # Test with librosa backend
        pitch_librosa = PitchShift(
            min_semitones=-2.0,
            max_semitones=2.0,
            p=1.0,
        )
        result = pitch_librosa(samples=test_audio, sample_rate=sample_rate)
        print("✓ PitchShift with librosa backend works")

    except Exception as e:
        print(f"✗ PitchShift with librosa backend failed: {e}")
        traceback.print_exc()

    try:
        # Test with signalsmith backend (default)
        pitch_signal = PitchShift(
            min_semitones=-2.0,
            max_semitones=2.0,
            p=1.0
        )
        result = pitch_signal(samples=test_audio, sample_rate=sample_rate)
        print("✓ PitchShift with signalsmith backend works")

    except Exception as e:
        print(f"✗ PitchShift with signalsmith backend failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    check_installation()
    test_individual_transforms()
    test_compose()
    inspect_pitch_shift()
    test_pitch_shift_backends()