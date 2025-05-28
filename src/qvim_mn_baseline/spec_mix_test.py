import torchaudio
import torch
import matplotlib.pyplot as plt
import librosa
from datasets import load_dataset
import random
import torch
import numpy as np
from specmix import specmix


speech_commands_v1 = speech_commands_v1["train"].train_test_split(
    train_size=0.5, test_size=0.5, stratify_by_column="label"
)

speech_commands_v1 = speech_commands_v1.filter(
    lambda x: x["label"]
    != (
        speech_commands_v1["train"].features["label"].names.index("_unknown_")
        and speech_commands_v1["train"].features["label"].names.index("_silence_")
    )
)

speech_commands_v1["train"] = speech_commands_v1["train"].select(
    [i for i in range((len(speech_commands_v1["train"]) // BATCH_SIZE) * BATCH_SIZE)]
)
speech_commands_v1["test"] = speech_commands_v1["test"].select(
    [i for i in range((len(speech_commands_v1["test"]) // BATCH_SIZE) * BATCH_SIZE)]
)

print(speech_commands_v1)


labels = speech_commands_v1["train"].features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label

print(id2label)

df = speech_commands_v1['train'].to_pandas()

audios = []
labels = []
for i in range(100, 500, 5):
    audio, sr = torchaudio.load(df['file'][i])
    if audio.shape != (1, 16000):
        continue
    label = df['label'][i]
    audios.append(audio)
    labels.append(label)

audios = torch.cat(audios)
labels = torch.tensor(labels)
spec_extractor = torchaudio.transforms.Spectrogram()
specs = spec_extractor(audios)
one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=- 1)
specs_original = torch.clone(specs)
out = specmix(specs[0:50], one_hot_labels[0:50], 0.99, 20, 30, 4, 3)

def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
# https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html
  fig, axs = plt.subplots(1, 1)
  axs.set_title(title or 'Spectrogram (db)')
  axs.set_ylabel(ylabel)
  axs.set_xlabel('frame')
  im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
  if xmax:
    axs.set_xlim((0, xmax))
  fig.colorbar(im, ax=axs)
  plt.show(block=False)


plot_spectrogram(specs_original[0])
plot_spectrogram(out[0][0])
out[1][0]
labels[0]
labels[49]