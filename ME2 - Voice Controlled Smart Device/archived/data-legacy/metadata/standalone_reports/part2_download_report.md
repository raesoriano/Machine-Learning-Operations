# Part 2 — Dataset Download Report

_Generated: 2026-09-21 13:20 UTC_

Raw speech datasets were downloaded from their official sources into `data/raw/`.
Originals are preserved unmodified (no extraction, no preprocessing).
Checksums (MD5) were verified against the values published by each source where available.

## Summary

- Datasets downloaded successfully: **3** of 3
- Datasets skipped/failed: **0**
- Total disk space used (raw downloads): **10.00 GB** (10,732,510,070 bytes)

## Per-dataset detail

### SLURP (audio + textual annotations)

- **Status:** ok
- **Source:** https://github.com/pswietojanski/slurp
- **Version/date:** EMNLP 2020 release; audio via Zenodo record 4274930 (2020-11-11)
- **Location:** `data/raw/slurp/ (repo/ = git clone, audio/ = Zenodo audio)`
- **Repo commit:** `8eb16545762be97ace75334109d73824217311f1` (25 files)
- **Total size:** 6.29 GB (6,757,856,494 bytes)

| File | Size | MD5 | MD5 verified | Archive entries |
|---|---|---|---|---|
| `data/raw/slurp/audio/slurp_real.tar.gz` | 3.65 GB | `9efc0f058ced…` | ✅ yes | 72,396 |
| `data/raw/slurp/audio/slurp_synth.tar.gz` | 2.64 GB | `6186b4d2475a…` | ✅ yes | 69,258 |
| `data/raw/slurp/audio/LICENSE.txt` | 0.00 GB | `eb6f866654ad…` | ✅ yes | - |

### Fluent Speech Commands

- **Status:** ok
- **Source:** https://zenodo.org/records/11106540
- **Version/date:** Zenodo record 11106540, DOI 10.5281/zenodo.11106540, published 2024-05-02, CC-BY-4.0
- **Location:** `data/raw/fluent_speech_commands/`
- **Total size:** 1.44 GB (1,545,730,387 bytes)

| File | Size | MD5 | MD5 verified | Archive entries |
|---|---|---|---|---|
| `data/raw/fluent_speech_commands/fluentai.zip` | 1.44 GB | `625d5dfecef8…` | ✅ yes | 30,154 |

### Google Speech Commands v2

- **Status:** ok
- **Source:** https://www.tensorflow.org/datasets/catalog/speech_commands
- **Version/date:** v0.02 official tarball (storage.googleapis.com/download.tensorflow.org)
- **Location:** `data/raw/google_speech_commands_v2/`
- **Total size:** 2.26 GB (2,428,923,189 bytes)

| File | Size | MD5 | MD5 verified | Archive entries |
|---|---|---|---|---|
| `data/raw/google_speech_commands_v2/speech_commands_v0.02.tar.gz` | 2.26 GB | `6b74f3901214…` | — (no published MD5) | 105,878 |

## Notes

- **Timers and Such (Zenodo record 4623772) was excluded** from this download set per
  project instruction. Its 13.1 GB archive was partially downloaded and then removed;
  it is not part of the canonical dataset pipeline.
- Google Speech Commands v2 has no officially published MD5; integrity was confirmed by
  archive validity (tar.gz lists cleanly) and entry count (105,878 entries incl. 105,835 WAV).
- SLURP audio is hosted on Zenodo (record 4274930) per the official GitHub repo's
  `scripts/download_audio.sh`; the textual annotations are the git clone under `data/raw/slurp/repo/`.
- No dataset was extracted, merged, or modified. Original archives are kept as downloaded.
