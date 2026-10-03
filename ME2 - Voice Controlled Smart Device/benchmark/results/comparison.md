| metric | v3 (PocketSphinx ensemble) | v6 (from-scratch HMM) | v7 (v3 arch, AM on v6 data) | v8 (Conformer+CTC, base) | v8 + negatives (reject) | v8 (736-vocab, content reject) |
|---|---|---|---|---|---|---|
| clips | 202 | 202 | 202 | 202 | 202 | 202 |
| intent acc (19) | 82.7% | 39.6% | 80.7% | 83.2% | 85.1% | 85.1% |
| intent acc 95% CI | [77-87%] | [33-46%] | [75-86%] | [77-88%] | [80-89%] | [80-89%] |
| intent balanced acc | 83.1% | 31.1% | 80.9% | 85.6% | 86.9% | 86.2% |
| intent macro F1 | 80.1% | 29.1% | 79.2% | 80.0% | 83.9% | 83.1% |
| intent macro F2 | 81.6% | 29.9% | 79.7% | 82.6% | 85.1% | 84.5% |
| false accept (OOS fired) | 87.5% (14/16) | 100.0% (16/16) | 81.2% (13/16) | 100.0% (16/16) | 62.5% (10/16) | 87.5% (14/16) |
| false reject (cmd ignored) | 2.7% | 0.0% | 3.2% | 0.0% | 1.6% | 2.2% |
| misfire (wrong cmd) | 8.6% | 57.0% | 10.8% | 9.7% | 9.1% | 6.5% |
| command acc (93) | 80.7% | 26.2% | 78.7% | 78.2% | 80.2% | 81.7% |
| command balanced acc | 85.8% | 28.2% | 83.2% | 84.0% | 83.4% | 86.8% |
| command macro F1 | 84.1% | 29.7% | 82.8% | 82.5% | 83.1% | 86.4% |
| command macro F2 | 84.4% | 27.8% | 82.0% | 82.6% | 82.6% | 86.1% |
| command misfire | 10.8% | 71.5% | 12.9% | 15.1% | 14.5% | 10.2% |
| slot exact (intent right) | 96.0% | 57.8% | 95.8% | 89.8% | 89.6% | 93.1% |
| latency p50 (s) | 0.12 | 8.68 | 0.14 | 0.04 | 0.04 | 0.04 |
| latency p95 (s) | 0.23 | 19.19 | 0.26 | 0.07 | 0.07 | 0.07 |
| real intent acc | 80.2% | 24.0% | 81.2% | 72.9% | 75.0% | 77.1% |
| synthetic intent acc | 84.9% | 53.8% | 80.2% | 92.5% | 94.3% | 92.5% |
| real command acc | 79.2% | 9.4% | 80.2% | 67.7% | 69.8% | 72.9% |
| synthetic command acc | 82.1% | 41.5% | 77.4% | 87.7% | 89.6% | 89.6% |
