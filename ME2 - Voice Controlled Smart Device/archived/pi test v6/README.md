# pi test v6 — from-scratch HMM/GMM voice-command listener

The same device loop as v3/v4/v5, but the recognizer is **built from scratch**
in numpy as a classic three-component Hidden-Markov-Model / Gaussian-Mixture-
Model system. It is *inspired by* PocketSphinx — same architecture, same three
pieces — but it does **not** use PocketSphinx (or any other ASR library):

```
"hey rhasspy"  ->  VAD capture  ->  acoustic model (waveform -> phone states)
                ->  phonetic dict (words -> phoneme strings)
                ->  language model  (constrains word sequences)
                ->  two-level Viterbi decode  ->  command  OR  REJECT
                ->  execute (music / weather / time / lights)  or
                    "I heard <words>. Can you repeat that?"
```

## The three components

| # | component | file(s) | what it does |
|---|-----------|---------|--------------|
| 1 | **Acoustic model (HMM/GMM)** | `hgm/acoustic.py`, `hgm/gmm.py`, `hgm/hmm.py` | one 3-state left-to-right HMM per phone, each state a 4-component diagonal GMM over 120-d features. Trained by bootstrap EM (Viterbi re-alignment + GMM EM). Converts the raw waveform into per-frame, per-phone, per-state log-likelihoods. |
| 2 | **Phonetic dictionary** | `hgm/dict.py` | the mapping file: each command word -> its ARPAbet phone sequence (from the CMU dictionary), i.e. words -> structural phoneme strings. |
| 3 | **Language model** | `hgm/lm.py` | word bigram + a constrained finite-state automaton over the 104-phrase grammar. Constrains the search space to valid word sequences. |

Decoding is a **two-level Viterbi** over the joint state
`(grammar-FSA node, phone position, HMM state)` (`hgm/decode.py`):

- the best path through the **grammar FSA** = the recognized command phrase;
- the best path through a **free unigram FSA** = "what I actually heard";
- the log-likelihood **gap** between the two drives **REJECT** — a non-command
  utterance decodes poorly under the grammar, so it is rejected instead of
  force-fit to the nearest command.

Silence is modeled explicitly: the FSA's start/end nodes are SIL regions driven
by a dedicated SIL HMM trained on real silence, so leading/trailing silence is
handled without padding the dictionary.

## Why v6 (vs v3 and v5)

| | v3 | v5 | **v6** |
|---|---|---|---|
| recognizer | PocketSphinx (C library) | 1-D CNN ASR + CNN classifier (ONNX) | **from-scratch HMM/GMM (numpy)** |
| acoustic model | PocketSphinx HMM/GMM (opaque) | 1-D CNN (opaque) | **3-state HMM x GMM, trained here** |
| phonetic dict | CMU dict (via PocketSphinx) | word vocab (no phones) | **word -> ARPAbet phones** |
| language model | JSGF grammar | phrase classifier | **grammar FSA (constrained Viterbi)** |
| rejection | none (forced) | learned REJECT class | **grammar-vs-free score gap** |
| new pip deps | pocketsphinx | onnxruntime | **none (numpy + scipy)** |
| model footprint | ~8 MB | ~1 MB | **~0.9 MB (acoustic npz)** |

v6 keeps the two things v5 got right — *the transcript is what was actually
heard* and *it genuinely rejects* — but replaces the black-box deep models with
the classic, fully-inspectable HMM/GMM pipeline. Every component (features,
GMM, HMM, FSA, Viterbi) is a few hundred lines of readable numpy.

## The models

| file | what | size |
|---|---|---|
| `models/acoustic_model.npz` | 36 phone HMMs (3 states x 4-comp GMM) + SIL | ~0.9 MB |
| `models/dictionary.txt` | 104 words -> ARPAbet phones | ~2 KB |
| `models/lm.npz` | bigram + 104-phrase grammar FSA | ~8 KB |
| `models/phones.txt` | 37-phone inventory (incl. SIL) | <1 KB |

Feature pipeline (identical train / test / Pi): 16 kHz mono -> peak-norm 0.9 ->
40 log-mel bins (25 ms window, 10 ms hop, n_fft=512) -> + delta + delta-of-
delta -> (T, 120), per-clip mean-normalized. Pure numpy/scipy (`hgm/feats.py`),
so the Pi needs **no torch and no onnxruntime for the recognizer**.

## Training

```bash
python train_am.py     # builds dict + LM, trains the acoustic model (bootstrap EM)
```

Trains on the 50-speaker real command clips (ALARM present in this checkout)
plus the synthetic OptionB clips (full phone coverage for all 31 command
classes) plus real silence for the SIL model. ~15 min on the HPC box.

## Run it (Pi)

```bash
cd ~/Machine-Learning-Operations
git pull
cd "ME2 - Voice Controlled Smart Device/pi test v6"
pip install -r requirements.txt   # openwakeword + piper + sounddevice + yt-dlp
sudo apt install mpv              # one-time, for the music commands

python vcm_pi_v6.py --wake-check  # expect PASS ~0.6
python vcm_pi_v6.py               # live: say "hey rhasspy", then a command
```

Useful flags:

```bash
python vcm_pi_v6.py --file some.wav    # classify one clip, no mic
python vcm_pi_v6.py --test             # run the 176-clip held-out set
python vcm_pi_v6.py --speak-heard      # also speak "I heard: ..."
python vcm_pi_v6.py --music-test       # verify the music pipeline
python vcm_pi_v6.py --no-play          # print responses, don't play
```

## Evaluation

`eval_v6.py` runs the 176-clip held-out set (19 command classes x 9 + 5
Tagalog REJECT clips) with the same ground truth and metric definitions as v3,
so the numbers are directly comparable. Report -> `test_v6_report.json`.

## Files

- `vcm_pi_v6.py` — the listener (wake word -> VAD -> HMM/GMM decode -> act).
- `hgm/` — the from-scratch recognizer: `feats`, `gmm`, `hmm`, `acoustic`,
  `dict`, `lm`, `decode`, `grammar`, `commands`.
- `train_am.py` — builds dict + LM and trains the acoustic model.
- `eval_v6.py` — held-out-set evaluation (v3-compatible metrics).
- `models/` — the trained acoustic model, dictionary, LM, phone inventory.
- `vcm_commands_enh3.jsgf` — the 104-phrase grammar.
- `wakeword/`, `responses/`, `vcm2/` — carried over from v3/v5.
