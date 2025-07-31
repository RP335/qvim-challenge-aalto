# AES AIMLA Challenge 2025 System by Team AudioAlchemy
## Query-by-Vocal Imitation Challenge

*Query by Vocal Imitation* (QVIM) enables users to search a database of sounds via a vocal imitation of the desired sound.
This offers sound designers an intuitively expressive way of navigating large sound effects databases. 


This is our code for this challenge. 

**Important Dates**
- Challenge start: April 1, 2025 
- Challenge end: June 15, 2025
- Challenge results announcement: July 15, 2025

For more details, please have a look at our [website](https://qvim-aes.github.io/#portfolio).


## Baseline System
This repository contains the modified baseline system for the AES AIMLA Challenge 2025. 
The architecture and the training procedure is based on ["Improving Query-by-Vocal Imitation with Contrastive Learning and Audio Pretraining"](https://dcase.community/documents/workshop2024/proceedings/DCASE2024Workshop_Greif_36.pdf) (DCASE2025 Workshop). 

* The training loop is implemented using [PyTorch](https://pytorch.org/) and [PyTorch Lightning](https://lightning.ai/). 
* Logging is implemented using [Weights and Biases](https://wandb.ai/site).
* It uses the [MobileNetV3](https://arxiv.org/abs/2211.04772) (MN) pretrained on AudioSet to encode audio recordings.
* The system is trained on [VimSketch](https://interactiveaudiolab.github.io/resources/datasets/vimsketch.html) and evaluated on the public evaluation dataset described on our [website](https://qvim-aes.github.io/#portfolio).


## Getting Started

Prerequisites
- linux (tested on Ubuntu 24.04)
- [conda](https://www.anaconda.com/docs/getting-started/miniconda/install), e.g., [Miniconda3-latest-Linux-x86_64.sh](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh)


1. Clone this repository.

```
git clone https://github.com/qvim-aes/qvim-baseline.git
```

2. Create and activate a conda environment with Python 3.10:
```
conda create -f environment.yml
conda activate qvim-ensemble
```

3. Install 7z, e.g., 

```
# (on linux)
sudo apt install p7zip-full
# (on windows)
conda install -c conda-forge 7zip
```
*For linux users*: do not use conda package p7zip - this package is based on the outdated version 16.02 of 7zip; to extract the dataset, you need a more recent version of p7zip.

4. If you have not used [Weights and Biases](https://wandb.ai/site) for logging before, you can create a free account. On your
machine, run ```wandb login``` and copy your API key from [this](https://wandb.ai/authorize) link to the command line.


## Training

All training is handled by the unified script `src/qvim_mn_baseline/train.py`. It is highly configurable and supports multiple model architectures and data augmentation strategies.

To see all available options, run:

```bash
export PYTHONPATH=$(pwd)/src
python src/qvim_mn_baseline/train.py --help
```

### Usage Examples

#### 1\. Train the MobileNetV3 Baseline (Original)

This command replicates the original baseline without advanced augmentations.

```bash
python src/qvim_mn_baseline/train.py \
    --model_type mobilenet \
    --project "qvim-experiments" \
    --model_save_path "checkpoints"
```

#### 2\. Train a PaSST Model with "Light" Augmentations

This example uses the powerful PaSST model and enables the "light" augmentation profile.

```bash
python src/qvim_mn_baseline/train.py \
    --model_type passt \
    --use_augmentations true \
    --aug_profile light \
    --batch_size 12 \
    --n_epochs 50 \
    --project "qvim-experiments" \
    --model_save_path "checkpoints"
```

#### 3\. Train a BEATs Model with "Full" Augmentations and SpecMix

This command trains the BEATs model using the "full" augmentation profile, which includes SpecMix.

```bash
python src/qvim_mn_baseline/train.py \
    --model_type beats \
    --beats_checkpoint_path "path/to/your/BEATs_iter3.pt" \
    --use_augmentations true \
    --aug_profile full \
    --batch_size 8 \
    --n_epochs 75 \
    --project "qvim-experiments" \
    --model_save_path "checkpoints"
```

## Evaluation Results


| Model Name      | MRR (exact match) | 
| --------------- | ----------------- | 
| random          | 0.0444            | 
| MN baseline     | 0.2726            | 
| *MN + Light Aug* | 0.2835           | 
| *PaSST* | 0.1502           | 
| *PANNs* | 0.1577          | 
| *BEATs* | 0.2309         | 



## Contact
For questions or inquiries, please contact [rahul.peter@aalto.fi](mailto:rahul.peter@aalto.fi) or [vivek.mohan@aalto.fi](mailto:vivek.mohan@aalto.fi).


## Citation

```
@inproceedings{Greif2024,
    author = "Greif, Jonathan and Schmid, Florian and Primus, Paul and Widmer, Gerhard",
    title = "Improving Query-By-Vocal Imitation with Contrastive Learning and Audio Pretraining",
    booktitle = "Proceedings of the Detection and Classification of Acoustic Scenes and Events 2024 Workshop (DCASE2024)",
    address = "Tokyo, Japan",
    month = "October",
    year = "2024",
    pages = "51--55"
}
```
