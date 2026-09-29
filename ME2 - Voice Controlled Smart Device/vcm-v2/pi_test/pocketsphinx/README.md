# PocketSphinx command-grammar fallback (~10.75 MB)

A self-contained, small-footprint voice-command recognizer for the Pi,
using the **stock** PocketSphinx `en-us` acoustic model + a JSGF command
grammar constrained to the 93 VCM command phrases. This is the **fallback**
for when the custom-trained acoustic model (see `backbone/pocketsphinx/`)
isn't available or is too large for the target device.

## Contents
| File | Size | Purpose |
|---|---|---|
| `en-us/` | 6.6 MB | Stock en-us acoustic model (mdef, means, variances, sendump, ...) |
| `cmudict-en-us.dict` | 3.3 MB | Word→phone dictionary |
| `en-us-phone.lm.bin` | 0.86 MB | Phone-level LM (enough for JSGF-constrained decode) |
| `vcm_commands.jsgf` | 2.6 KB | The 93 command phrases (grammar) |

**Total: ~10.75 MB** (vs 290 MB for the Whisper ONNX).

## Performance (171-clip sole test set, one new speaker)
| Command | Intent | WER | Latency p50 |
|---|---|---|---|
| 78.4% | 83.0% | 0.31 | 61 ms |

This *is* the stock model + grammar, so the numbers match the stock
command-grammar baseline exactly.

## Run it
```python
from pocketsphinx import Config, Decoder
import wave, os
base = os.path.dirname(__file__)
cfg = Config()
cfg.set_string("-hmm", os.path.join(base, "en-us"))
cfg.set_string("-dict", os.path.join(base, "cmudict-en-us.dict"))
cfg.set_string("-lm", os.path.join(base, "en-us-phone.lm.bin"))
cfg.set_string("-logfn", "/dev/null")
dec = Decoder(cfg)
with open(os.path.join(base, "vcm_commands.jsgf")) as f:
    dec.add_jsgf_string("vcm", f.read())
dec.activate_search("vcm")
# feed 16 kHz mono 16-bit PCM:
dec.start_utt()
with wave.open("clip.wav", "rb") as w:
    while True:
        fr = w.readframes(4000)
        if not fr:
            break
        dec.process_raw(fr, False, False)
dec.end_utt()
print(dec.hyp().hypstr)  # e.g. "alarm six am"
```

Then feed the transcript to the stage-2 classifier to get the command/intent,
exactly as the Whisper pipeline does. See `backbone/pocketsphinx/eval_pocketsphinx.py`
for the full eval (same ground truth + classifier + WER definition).

## Why a fallback?
The **custom-trained** acoustic model now **beats** this stock baseline:
`backbone/artifacts/pocketsphinx_trained_lda_enh/` (1.6 MB, 200-senone LDA AM
+ 102-phrase JSGF) scores **87.1% command / 90.1% intent** on the 171-clip
held-out set, vs 78.4% here — at 1/6th the footprint and 2x lower latency.
It is the primary ultra-small recognizer. This stock en-us + JSGF folder is
kept as the **guaranteed zero-training** baseline that always works (no
sphinxtrain build needed), and as a reference for the grammar format.
