# Machine Exercise 1 — 3-Layer CNN for MNIST with einops / einsum

A 3-convolution-layer CNN for MNIST digit classification in which **every layer
and tensor operation is implemented with `einops` / `torch.einsum`**:

| Operation        | Standard PyTorch | This notebook |
|------------------|------------------|---------------|
| Conv2d           | `F.conv2d`       | `Tensor.unfold` (im2col) → `einops.rearrange` → `torch.einsum` |
| MaxPool2d (2×2)  | `F.max_pool2d`   | `einops.reduce(..., 'max')` |
| Flatten          | `x.view(-1)`     | `einops.rearrange('b c h w -> b (c h w)')` |
| Linear (FC)      | `F.linear`       | `torch.einsum('bn,ne->be')` |

## Architecture

Input `(b, 1, 28, 28)`:

```
Conv2d(1→32, 3×3, pad 1) → ReLU → MaxPool2  → (b, 32, 14, 14)
Conv2d(32→64, 3×3, pad 1) → ReLU → MaxPool2 → (b, 64, 7, 7)
Conv2d(64→128, 3×3, pad 1) → ReLU → Flatten → (b, 6272)
Linear(6272→10) → logits
```

## Run

Use the repository environment (see the top-level `README.md` / `setup_env.sh`),
then run all cells of `ME1_einops_einsum_cnn.ipynb` top to bottom.

```bash
jupyter notebook ME1_einops_einsum_cnn.ipynb
```

Training runs on GPU when available (`device = cuda`); on this cluster that is
an NVIDIA A100-40GB. 5 epochs of MNIST take well under a minute.

## Results

- Training: 5 epochs, Adam (lr=1e-3), batch size 64, MNIST (60k train / 10k test)
- **Test-set accuracy: 0.9892** (9892/10000)
- 16 random test samples shown in a 4×4 grid with ground-truth vs. prediction
  (also saved as `grid_16.png`)

## Files

- `ME1_einops_einsum_cnn.ipynb` — the exercise (executed, with outputs)
- `grid_16.png` — the 4×4 prediction grid
- `build_me1.py` / `run_me1.py` — scripts that (re)generate and execute the notebook
