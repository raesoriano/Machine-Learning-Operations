# Part 1 — Project Setup Report

Date: 2026-09-21

## Objective

Set up the project structure for a Raspberry Pi 5 standalone voice-command
recognition project (local speech-command classifier, no cloud, no LLM).
No datasets downloaded and no model trained in this part.

## Files / directories created

```
Standalone-Voice-Command-Recognition-Project/
├── README.md                    # project overview, pipeline, structure
├── configs/
│   └── commands.yaml            # canonical command labels (single source of truth)
├── data/
│   ├── raw/                     # downloaded datasets, untouched
│   ├── interim/                 # temporary processing artifacts
│   ├── processed/               # canonical dataset
│   └── metadata/                # manifests, stats, download records
├── scripts/                     # entry-point scripts (download/process/train/eval/export)
├── src/
│   ├── data/                    # dataset loading and canonicalization
│   ├── features/                # audio feature extraction
│   ├── models/                  # model definitions
│   ├── training/                # training loops and utilities
│   └── inference/               # Raspberry Pi inference and export
├── checkpoints/                 # saved model weights
├── experiments/                 # per-run logs, metrics, artifacts
├── reports/                     # setup and progress reports (this file)
└── tests/
```

Empty directories are tracked via `.gitkeep` files.

## Canonical labels

Defined in `configs/commands.yaml` (11 classes):

| Index | Label         |
|-------|---------------|
| 0     | yes           |
| 1     | no            |
| 2     | up            |
| 3     | down          |
| 4     | left          |
| 5     | right         |
| 6     | on            |
| 7     | off           |
| 8     | stop          |
| 9     | go            |
| 10    | _background   |

The label order in the file defines the class index used for training and
inference.

## Python environment

| Item                  | Value                                              |
|-----------------------|----------------------------------------------------|
| Interpreter           | `/opt/miniconda3/bin/python3` (Miniconda)          |
| Python version        | 3.13.9                                             |
| pip                   | 25.2                                               |
| OS                    | Linux 6.8.0-124-generic, x86_64                    |
| CPU                   | AMD EPYC 7742 64-Core (256 logical CPUs)           |
| RAM                   | ~1008 GB                                           |
| Free disk             | ~372 GB                                            |

## PyTorch / CUDA / GPU

| Item            | Value                                            |
|-----------------|--------------------------------------------------|
| PyTorch version | **not installed** (missing)                      |
| CUDA (torch)    | n/a (torch not installed)                        |
| GPU             | 8× NVIDIA A100-SXM4-40GB (40960 MiB each)        |
| NVIDIA driver   | 580.159.03 (supports CUDA 13.0)                  |

The build machine has 8× A100 GPUs, but PyTorch is not yet installed, so no
CUDA build is available to Python at this time.

## Installed packages (relevant)

| Package        | Version |
|----------------|---------|
| numpy          | 2.5.2   |
| scipy          | 1.18.1  |
| pandas         | 3.0.5   |
| matplotlib     | 3.11.1  |
| PyYAML         | 6.0.3   |
| scikit-learn   | 1.9.1   |
| pillow         | 12.3.0  |

## Missing dependencies

| Package    | Needed for                              |
|------------|------------------------------------------|
| torch      | model training and export (Part 4)       |
| torchaudio | audio loading / resampling               |
| soundfile  | WAV I/O for dataset processing           |
| librosa    | audio feature extraction (optional)      |
| pytest     | test suite (Part 6+)                     |

These are intentionally **not** installed yet — they will be added in the
parts where they are first needed, keeping the environment minimal.

## Notes

- Config format is YAML (PyYAML already available; no extra dependency).
- The target deployment device is a Raspberry Pi 5 (ARM64, CPU-only); GPU
  availability on this build machine is only for training.
- No datasets downloaded, no model trained, per task constraints.
