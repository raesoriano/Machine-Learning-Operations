# ME2 — 5–10 minute live demo script

A timed run-of-show for a quick live demo of the ME2 voice-controlled smart
device. Pairs the **live mic demo** (`live_demo/`) with the **benchmark**
numbers and the **results deck** (`me2-deck.md`). Everything runs on-device —
no cloud, no LLM.

> **Before you start:** have `live_demo/` set up (see its README), the v8 ONNX
> loaded, a speaker + mic, and `me2-deck.md` / `benchmark/results/comparison.md`
> open. Rehearse the wake word once so the threshold is right.

---

## 0:00 — Hook (30 s)

> "This is a voice-controlled smart device that runs **entirely on a
> Raspberry Pi 5** — no cloud, no phone, no LLM. I say a wake word, give a
> command, and it acts. The whole recognizer is a **10.9-million-parameter
> Conformer, 43 MB, exported to ONNX**, and it decodes a command in **~40
> milliseconds**."

*(Open `me2-deck.md` to the Model block.)*

## 0:30 — The model in one picture (60 s)

Walk the pipeline on the deck:

> "16 kHz mic → log-mel → a **causal** 6-layer Conformer (it only looks left,
> so it streams) → a **CTC** head over a **736-word** vocabulary → a
> constrained decoder over the **93 command phrases** → **19 intents** and
> **18 slots**. The key trick for rejection: I *expanded the vocabulary* so
> out-of-scope speech decodes to **real words** — and then I reject it because
> those words don't form any of the 93 commands. It's rejected, not
> force-guessed."

## 1:30 — LIVE: in-scope commands (2–3 min)

Run `python vcm_live.py --model v8`. Do **4–6 commands** across different
intents (mix a slotted one to show the slot head):

1. "hey rhasspy" → **"play some music"** → it starts the playlist.
2. "hey rhasspy" → **"turn the lights on"** → lights response.
3. "hey rhasspy" → **"set the brightness to 60 percent"** → *(slot: 60 %)*.
4. "hey rhasspy" → **"what's the weather"** → live weather answer (Piper TTS).
5. "hey rhasspy" → **"set a timer for 30 seconds"** → *(slot: 30 s)*.
6. "hey rhasspy" → **"volume up"** → volume step.

> (After each: "That's the wake word arming the mic, the 'yes?' cue, the
> command decoding, and the spoken response — all on the Pi.")

## 4:30 — LIVE: rejection (60 s)

Show the model **refusing** things that aren't commands:

1. "hey rhasspy" → **"what's the capital of France"** → "can you repeat that?"
2. "hey rhasspy" → *(just noise / "um, the, yeah")* → rejected.

> "This is the part that used to be broken. Older versions either **force-mapped**
> nonsense to a command, or **rejected real commands**. The expanded vocabulary
> fixed the false-rejects — real commands now decode cleanly — and the content
> rule rejects the rest."

## 5:30 — The numbers (90 s)

Open `benchmark/results/comparison.md`. Two quick points:

> "We benchmarked every model we tried on the same 202-clip held-out set, with
> the **official vcm-benchmark** scoring. The current v8 is the best on every
> axis that matters:
> - **85.1 % intent, 81.7 % command, 93.1 % slot exact.**
> - **10.2 % misfire** — the lowest.
> - **~40 ms** decode — vs 120 ms for the PocketSphinx baseline and **8.7
>   seconds** for the from-scratch HMM.
>
> The honest caveat: **false-accept is 87.5 %** on the 16 out-of-scope clips —
> because OOS now decodes to real words, half of which contain a command word.
> That's the known trade-off of the vocabulary expansion; it's why real-voice
> accuracy (52 %) is the open problem, and synthetic is 99 %."

## 7:00 — What's in the repo (60 s)

> "Everything is in one public repo: the **model** (train/eval/ONNX in
> `pi test v8-conformer-ctc/`), the **dataset** (Hugging Face, one-command
> download), the **benchmark** (offline, with the committed cluster results),
> and this **live demo**. The README has a **one-command** build — clone,
> download, train on 3 A100s in ~11 minutes, eval, export, benchmark. And the
> whole thing runs on the Pi with just `onnxruntime` and `numpy`."

## 8:00 — Close (30 s)

> "So: always-on wake word, intent + slot recognition, rejection that doesn't
> guess, all at the edge in ~40 ms, 43 MB. That's ME2."

*(Optional 30 s Q&A buffer to reach 10 min.)*

---

### Quick reference (cheat sheet)

| moment | show | say the key number |
|---|---|---|
| model | `me2-deck.md` Model block | 10.9 M params · 43 MB · 736 words · causal/streaming |
| live in-scope | `vcm_live.py --model v8` | wake → cue → command → speak, on-device |
| live reject | same | rejected, not force-guessed |
| numbers | `benchmark/results/comparison.md` | 85.1 % intent · 81.7 % cmd · 93.1 % slot · 10.2 % misfire · ~40 ms |
| caveat | same | 87.5 % false-accept (OOS) · real 52 % vs synth 99 % |
| repo | README one-go block | one-command build · 3×A100 ~11 min · Pi needs only onnxruntime |
