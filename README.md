# Machine Learning Operations

AI 231 — machine exercises repository. All machine exercises for the course live here.

## Environment

Dedicated conda environment: **`ai231`** (Python 3.11)

```bash
conda activate ai231
```

Installed from the requirements of
[roatienza/Deep-Learning-Experiments](https://github.com/roatienza/Deep-Learning-Experiments)
(see `requirements.txt` in this repo):

- PyTorch stack: `torch`, `torchvision`, `torchaudio`, `accelerate`
- Data/viz: `numpy`, `scipy`, `Pillow`, `matplotlib`, `scikit-image`
- Audio: `librosa`
- Models: `timm`, `einops`, `lightning`, `transformers`, `sentencepiece`
- Serving/logging: `gradio`, `wandb`

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
└── requirements.txt
```
