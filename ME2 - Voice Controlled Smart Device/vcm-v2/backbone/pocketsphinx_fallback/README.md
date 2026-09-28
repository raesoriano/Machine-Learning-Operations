# PocketSphinx fallback model (9.5 MB)

Stock `en-us` acoustic model + cmudict + the VCM command grammar (JSGF).
This is the small-footprint fallback for the Pi: **78.4% command / 83.0% intent**
on the 171-clip sole test set, ~60 ms decode latency, 9.5 MB total — vs the
290 MB Whisper fine-tune (85.4% / 86.0%).

## Contents

| File | What |
|---|---|
| `en-us/` | Acoustic model (HMMs: `means`, `variances`, `transition_matrices`, `mdef`, `sendump`, `noisedict`, `feat.params`) |
| `dict` | cmudict-en-us word->phone dictionary |
| `vcm_commands.jsgf` | Command grammar (93 phrases) — the key piece; without it, free dictation scores 17.5% |

## Usage (python pocketsphinx)

    import pocketsphinx

    dec = pocketsphinx.Decoder(
        hmm="backbone/pocketsphinx_fallback/en-us",
        dictionary="backbone/pocketsphinx_fallback/dict",
    )
    dec.add_jsgf_string("vcm", open("backbone/pocketsphinx_fallback/vcm_commands.jsgf").read())
    dec.set_grammar("vcm")
    dec.start_utt()
    dec.process_raw(wav_bytes, full=True)
    dec.end_utt()
    print(dec.hyp().hypothesis)   # e.g. "turn on the lights"

Notes:
- Feed 16 kHz 16-bit mono PCM (the ME2 clips are already 16 kHz).
- No LM file is needed in grammar mode — the JSGF grammar replaces the bigram LM.
- This is the *stock* AM. A custom AM trained on our 7,164 clean clips is the
  planned upgrade (see `backbone/pocketsphinx/build_trained_am.py`); it will
  replace `en-us/` + `dict` in place, same layout.

## How it was evaluated

`backbone/pocketsphinx/eval_pocketsphinx.py --hmm .../en-us --dict .../dict
--jsgf backbone/pocketsphinx/vcm_commands.jsgf` over
`test_data/additional_test_data` (171 clips). Report:
`backbone/reports/pocketsphinx_stock_cmd.json`.
