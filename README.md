# Machine Learning Operations

Repository for machine learning exercises. Each exercise lives in its own
folder (`ME1/`, `ME2/`, ...).

## Environment

A dedicated conda environment is used for all exercises. It is built from
`requirements.txt` in this repo, which mirrors the requirements of
[roatienza/Deep-Learning-Experiments](https://github.com/roatienza/Deep-Learning-Experiments):

- PyTorch stack: `torch`, `torchvision`, `torchaudio`, `accelerate`
- Data/viz: `numpy`, `scipy`, `Pillow`, `matplotlib`, `scikit-image`
- Audio: `librosa`
- Models: `timm`, `einops`, `lightning`, `transformers`, `sentencepiece`
- Serving/logging: `gradio`, `wandb`

To rebuild the environment from scratch:

```bash
bash setup_env.sh
```

By default the environment is created in `./env` inside this repo. To place
it elsewhere (e.g. on a shared filesystem in a cluster setup), override the
location:

```bash
MLOPS_ENV_DIR=/path/to/env bash setup_env.sh
```

Then activate it:

```bash
conda activate /path/to/env
```

## Hardware

Exercises run on a **DGX cluster** node with **8× NVIDIA A100-SXM4-40GB**.
Use the GPUs for training (e.g. `CUDA_VISIBLE_DEVICES=0`, or multi-GPU via
`accelerate` / DDP).

## Layout

```
Machine-Learning-Operations/
├── ME1/            # Machine Exercise 1
├── ME2/            # Machine Exercise 2
├── ...
├── requirements.txt
└── setup_env.sh
```
