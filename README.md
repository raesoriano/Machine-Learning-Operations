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

> **ME2 (`pi test v3`) does NOT need this env.** The deployed voice-command
> model is torch-free (`pocketsphinx` + `scikit-learn` + `openwakeword`) and
> ships its own `ME2 - Voice Controlled Smart Device/pi test v3/requirements.txt`.
> On a Raspberry Pi, install only that file — never this root `requirements.txt`
> or `setup_env.sh` (they pull in a ~2 GB PyTorch stack). See the ME2 README.

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
