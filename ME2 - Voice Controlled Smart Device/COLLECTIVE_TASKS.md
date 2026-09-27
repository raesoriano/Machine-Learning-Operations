# Collective Tasks — Work Plan & Assignments

**Scope:** the two collective deliverables from PLAN.md —
**(A) the training dataset** and **(B) the VCM benchmark** — plus the shared
contracts that make individual work (model, validation, demo) plug in cleanly.

**Class size:** 35–40 students. This doc splits the work into 5 work packages
(WP) with named roles, task checklists, acceptance criteria, and a 6-week
timeline that matches PLAN.md §7.

**Already in the repo (do not rewrite, build on it):**

| Path | What it is | Who owns changes |
|------|-----------|------------------|
| `vcm/slot_space.py` | Closed slot values (locations, colors, contacts, genres, …) — single source of truth | WP1 only, via PR |
| `vcm/parser.py` | Deterministic transcript → `{intent, slots}` parser | WP1 (rules) + WP4 (scoring) |
| `vcm/vocab.py` | Constrained model vocabulary (built from slot_space) | auto from WP1 |
| `vcm/features.py` | log-mel recipe (train = serve, no drift) | model owner |
| `data/templates/grammar.py` | Per-intent template grammars (text + gold slots) | WP2 (one owner per intent) |
| `data/generate/tts_piper.py`, `augment.py` | TTS + augmentation pipeline (seeded, resumable) | WP3 |
| `benchmark/evaluate.py` | Harness: `--testset synthetic\|frozen  --model mock*\|onnx\|vosk` | WP4 |
| `benchmark/metrics.py` | intent acc, slot F1, exact match, WER, rejection, per-intent/subset | WP4 |
| `benchmark/synthetic_set.py` | Swappable stand-in test set (works today, no audio) | WP4 |
| `benchmark/frozen_set.py` | Frozen-set loader + schema validation (drop-in for v1) | WP4 |
| `tools/check_grammar.py` | CI gate: every generated row must round-trip through the parser | WP1/WP2 |

**The golden rule:** the parser is the contract. Dataset rows, benchmark gold
labels, model output, and the RPi runtime all flow through `vcm.parser.parse`.
If you change a rule or a slot value, `python -m tools.check_grammar` must
still pass 100%, and you must re-run the benchmark sanity check
(`python -m benchmark.evaluate --testset synthetic --model mock_clean` → 100%).

---

## 1. Team structure (35–40 people)

Five work packages, each with **one lead** (accountable, merges PRs) and
members. Leads form a 5-person "steering" group that meets twice a week (30
min) to resolve cross-WP conflicts (mostly: slot-space changes, benchmark
freeze date).

| WP | Name | Lead + members | Headcount |
|----|------|----------------|-----------|
| WP1 | Taxonomy & parser contract | 1 lead + 2 | 3 |
| WP2 | Template grammars (per intent) | 1 lead + 9 | 10 |
| WP3 | TTS + augmentation + manifests | 1 lead + 7 | 8 |
| WP4 | Benchmark set, harness, baselines, leaderboard | 1 lead + 9 | 10 |
| WP5 | QA, real-data subset, dataset card | 1 lead + 3 | 4 |
| — | Buffer / floaters (join whichever WP is behind) | — | 2–5 |

Total: 35 core + 2–5 floaters = 35–40. Everyone also does their individual
work (model / validation / demo) in parallel — collective work is ~40% of
effort, individual ~60%.

**Roster (fill in names):**

```
WP1 lead: ________   members: ________, ________
WP2 lead: ________   members (one per intent below):
    play_music: ________   ask_question: ________   lights_switch: ________
    lights_adjust: ________   set_timer: ________   set_alarm: ________
    set_temperature: ________   media_control: ________   reminders_lists: ________
    call: ________
WP3 lead: ________   members: ________, ________, ________, ________, ________, ________, ________
WP4 lead: ________   members: ________, ________, ________, ________, ________, ________, ________, ________, ________
WP5 lead: ________   members: ________, ________, ________
Floaters: ________, ________, ________, ________, ________
```

---

## 2. Work packages

### WP1 — Taxonomy & parser contract (3 people)

**Goal:** the command set, slot definitions, and parser are frozen early
enough that WP2–WP4 never block on them.

- [ ] Review the 10-intent taxonomy (PLAN.md §1) against the usage data;
      confirm intent names and slot schemas in `vcm/parser.py`.
- [ ] Review every closed slot list in `vcm/slot_space.py` (locations, colors,
      contacts, genres/artists, reminder actions, number ranges). Decide the
      final values **by end of week 1** — after that, changes need a lead
      sign-off because they break vocab + data + benchmark.
- [ ] Confirm the parser handles both digit and number-word forms ("5
      minutes" / "five minutes"), am/pm, C/F, and rejects OOV.
- [ ] Maintain `tools/check_grammar.py` as the CI gate; add a GitHub Actions
      workflow (`.github/workflows/check.yml`) that runs it on every PR.
- [ ] Document the contract: `vcm/README.md` — intent table, slot schema,
      examples, and the "how to add a slot value" PR checklist.

**Acceptance:** `check_grammar` green on main; contract doc merged; slot
space frozen (tag `slot-space-v1`).

---

### WP2 — Template grammars, one owner per intent (10 people)

**Goal:** every intent's generator in `data/templates/grammar.py` produces
natural, diverse, slot-balanced text whose gold labels match the parser
exactly.

Each intent owner:
- [ ] Owns their `gen_<intent>` function. Expands templates (≥ 8 phrasings
      per intent; e.g. "turn on the lights" / "lights on" / "switch on the
      bedroom lights").
- [ ] Guarantees slot coverage: every slot value in `slot_space` appears ≥ 20×
      in the generated corpus (add a coverage report:
      `python -m data.templates.grammar --out /tmp/x.jsonl --n 100000` then
      count — WP5 provides a one-liner script).
- [ ] Keeps gold slots **parser-consistent** (run `tools/check_grammar`
      locally before every PR).
- [ ] Adds ≥ 5 near-miss OOV utterances for their intent to
      `OOV_UTTERANCES` (e.g. for lights: "turn on the fan", "turn on the
      television") — each must parse to `unknown`.

The WP2 lead additionally:
- [ ] Balances intent frequencies (proportional to real usage: lights >
      play/music > media control > timers > …) via a `WEIGHTS` dict in
      `generate()`.
- [ ] Owns the OOV list review (dedupe, keep parser-consistent).

**Acceptance:** `check_grammar` 100% at n=20k; coverage report shows every
slot value ≥ 20×; OOV list ≥ 60 utterances, all rejected by the parser.

---

### WP3 — TTS + augmentation + manifests (8 people)

**Goal:** turn WP2's text manifests into a reproducible audio dataset
(~50–100k utterances) on shared storage, with a dataset card.

- [ ] **Voices (2):** install Piper + 1–2 extra open TTS engines on the
      generation machine; agree the voice list (20–40 voices, accent
      diversity) → `data/voices/piper_voices.yaml`.
- [ ] **Generation (2):** run `data/generate/tts_piper.py` over the train
      manifest (it's resumable — `--resume`); shard by intent so 2 machines
      can run in parallel; log per-row success/failure.
- [ ] **Augmentation (2):** run `data/generate/augment.py` with real RIRs
      (OpenSLR) + MUSAN/NOISEI; produce clean/noisy/far-field variants;
      verify SNR levels (5/10/15/20/25 dB) with a spot-check notebook.
- [ ] **Manifests & splits (1):** build speaker-disjoint 80/10/10
      train/val/test manifests in `data/manifests/` (JSONL, schema in
      `data/manifests/SCHEMA.md`); guarantee **no speaker overlap** between
      splits (script + report).
- [ ] **Dataset card (1):** `data/DATASET_CARD.md` — scale, voices,
      augmentations, splits, known limitations (TTS domain gap), license.

**Acceptance:** manifests validate (row count, split disjointness, audio
files exist); a 100-row random sample is manually listenable (WP5 spot-check);
dataset card merged.

---

### WP4 — Benchmark set, harness, baselines, leaderboard (10 people)

**Goal:** a frozen, versioned, swappable benchmark that any VCM can be
validated against with one command.

**Swappability contract (already implemented — preserve it):**
`benchmark/evaluate.py` takes `--testset synthetic|frozen` and
`--model mock_clean|mock_corrupt|onnx|vosk`. The frozen set is a JSONL
manifest (schema in `benchmark/frozen_set.py`); the model is anything with
`load(path)` + `transcribe(audio, sr) -> str`. **Swapping either axis must
require zero changes to `evaluate.py`.**

- [ ] **Frozen core set (3):** from the WP3 *test* split (speaker-disjoint
      from train), build `benchmark/testset/frozen_v1/core.jsonl` — ~2k
      in-vocab utterances, balanced per intent, gold labels = parser output
      on the gold transcript.
- [ ] **OOV set (1):** ~500 rejection utterances (from WP2's near-miss lists
      + new ones), all parser-verified `unknown`.
- [ ] **Noise + far-field subsets (2):** ~1k noisy (SNR 20/10/5 dB) and ~300
      far-field (RIR) rows, reusing the same core utterances with different
      audio variants → `subset` field = `noise`/`farfield`.
- [ ] **Harness hardening (1):** `evaluate.py` already does per-intent,
      per-subset, failure sampling. Add: confidence-interval reporting
      (bootstrap), and a `--limit` flag for quick runs.
- [ ] **Baselines (2):** (a) Vosk small + parser (the "off-the-shelf ASR +
      rules" line), (b) mock_corrupt at measured-WER levels. Record both in
      the leaderboard as floor references.
- [ ] **Leaderboard (1):** `benchmark/leaderboard.md` — table (model, date,
      intent acc, slot F1, exact, WER, rejection, RTF/RPi), seeded with the
      baselines. One PR per submission; the WP4 lead merges.
- [ ] **Freeze (lead):** tag `benchmark-v1` (git tag + manifest checksums in
      `frozen_v1/SHA256SUMS`). After the tag, the set is immutable; changes
      mean `frozen_v2` with a changelog.

**Acceptance:** `evaluate.py --testset frozen --testset-manifest
benchmark/testset/frozen_v1/manifest.jsonl --model vosk --model-path …` runs
end-to-end; leaderboard has ≥ 2 baseline rows; `benchmark-v1` tagged.

---

### WP5 — QA, real-data subset, dataset card review (4 people)

**Goal:** catch what automation misses; produce the human-verified slice.

- [ ] **Real-data recording (2):** record 100–500 utterances from 5–10
      people (phone/USB mic, different rooms); label with the parser
      (disagreements → WP1 ruling). This set is **validation-only, never
      trained on** — keep it in `data/real/` with its own manifest.
- [ ] **Spot-checks (1):** 500 random TTS rows — is the audio intelligible,
      does it match the text? Report defect rate; >5% defect → WP3 re-run.
- [ ] **Slot-coverage audit (1):** run the coverage script on the final
      train manifest; verify every slot value ≥ 20×; verify OOV list
      parser-consistency.
- [ ] **Dataset card review (all):** final pass on `data/DATASET_CARD.md`
      before it's declared done.

**Acceptance:** real-data manifest merged (validation split only); spot-check
report ≤ 5% defects; coverage audit green.

---

## 3. Collaboration rules

1. **Branch per task:** `wp2/lights-switch-templates`, `wp4/frozen-core-…`.
   Small PRs (< 300 lines) get faster review.
2. **CI gate on every PR touching `vcm/` or `data/templates/`:**
   `python -m tools.check_grammar` must pass 100%.
3. **CI gate on every PR touching `benchmark/`:**
   `python -m benchmark.evaluate --testset synthetic --model mock_clean` must
   print 100% (harness sanity).
4. **Slot-space changes (WP1) need a lead sign-off** — they cascade into
   vocab, data, and benchmark.
5. **No direct pushes to `main`.** The benchmark freeze is a git tag, not a
   folder rename.
6. **Audio never goes in git.** Audio lives on shared storage; git holds
   manifests (JSONL) + checksums.
7. **Status:** each WP lead updates the table below every Friday.

**Status board (updated Fridays):**

| WP | Week 1 | Week 2 | Week 3 | Week 4 | Week 5 | Week 6 |
|----|--------|--------|--------|--------|--------|--------|
| WP1 | | | | | | |
| WP2 | | | | | | |
| WP3 | | | | | | |
| WP4 | | | | | | |
| WP5 | | | | | | |

---

## 4. Timeline (matches PLAN.md §7)

| Week | WP1 | WP2 | WP3 | WP4 | WP5 |
|------|-----|-----|-----|-----|-----|
| 1 | **Freeze slot space** (`slot-space-v1`); CI workflow | Expand per-intent templates (≥ 8 phrasings each) | TTS prototype: 1 intent, 3 voices | Harness review; start core-set selection | Recording plan + consent form |
| 2 | Contract doc (`vcm/README.md`) | Weighted frequencies; OOV near-misses | Full TTS run (sharded); augmentation v1 | Core set draft; OOV set | Start real-data recording |
| 3 | Support WP4 label questions | Final grammar pass | Manifests + splits; speaker-disjoint report | **Freeze `benchmark-v1`**; baselines | Spot-checks (500 rows) |
| 4 | — | — | Dataset card v1 | Leaderboard v1 + hardening (CIs) | Coverage audit; card review |
| 5 | — | — | Re-runs if defects > 5% | Support individual benchmark runs | Real-data manifest merged |
| 6 | — | — | — | Final leaderboard | Final dataset card |

**Collective definition of done (by week 6):**
1. Slot space frozen + contract doc merged (WP1)
2. Dataset v1: manifests + splits + dataset card, coverage ≥ 20× per slot
   value (WP2+WP3)
3. Benchmark v1: frozen, tagged, harness one-command, ≥ 2 baselines on the
   leaderboard (WP4)
4. Real-data validation subset + spot-check report (WP5)

---

## 5. How individual work plugs in (for reference)

Individual students do **not** touch WP2/WP3/WP4 code. They consume:

- **Train:** `data/manifests/train_aug.jsonl` → `python -m model.train
  --manifest … --config model/configs/tiny.yaml --out model/checkpoints/tiny`
- **Validate:** `python -m benchmark.evaluate --testset frozen
  --testset-manifest benchmark/testset/frozen_v1/manifest.jsonl --model onnx
  --model-path model/checkpoints/vcm_v1_int8.onnx --report
  reports/<name>.json` → submit a leaderboard PR (WP4 lead merges).
- **Demo:** `deploy/rpi_service/server.py` (live mic or `--file`) +
  `demo/mock_home/server.py` dashboard; script in `demo/run_demo.py`.

If an individual finds a parser bug, they open an issue tagged `wp1` — they
do not fix it themselves (keeps the contract single-owner).
