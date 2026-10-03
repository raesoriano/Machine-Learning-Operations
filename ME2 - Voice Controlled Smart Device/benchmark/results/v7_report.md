> ME2 v7 - v3 architecture, custom AM retrained on the v6 dataset

# VCM benchmark - v7 - offline

Wake word: **hey rhasspy (offline: not used)** - trials: 202 with the wake word + 0 without - shuffle seed: n/a - connection: offline - holdout: airimonda/ai231-me2-voice-commands split=holdout (202 clips)

## At a glance

|                                   | overall       | real voice   | synthetic voice |
|-----------------------------------|---------------|--------------|-----------------|
| intent accuracy (19)              | 80.7%         | 81.2%        | 80.2%           |
| command accuracy (93)             | 78.7%         | 80.2%        | 77.4%           |
| false accept (out of scope fired) | 81.2% (13/16) | 70.0% (7/10) | 100.0% (6/6)    |
| false reject (command ignored)    | 3.2%          | 1.2%         | 5.0%            |
| false wake (no wake word, fired)  | -             | -            | -               |
| slot exact                        | 95.8%         | 97.6%        | 94.4%           |
| latency p95                       | 0.26 s        | 0.30 s       | 0.15 s          |

**Pi:** real-time factor 0.047 (p95 0.067), inference 142 ms

# Detailed metrics

## Classification

| metric                                       | 19 intents (+reject) | 93 commands (+reject) |
|----------------------------------------------|----------------------|-----------------------|
| accuracy                                     | 80.7%                | 78.7%                 |
| balanced accuracy                            | 80.9%                | 83.2%                 |
| precision (macro)                            | 81.5%                | 89.2%                 |
| recall (macro)                               | 80.9%                | 83.2%                 |
| F1 (macro)                                   | 79.2%                | 82.8%                 |
| F2 (macro)                                   | 79.7%                | 82.0%                 |
| false accept rate (OOS fired)                | 81.2%                | 81.2%                 |
| false reject rate (in-scope silent/rejected) | 3.2%                 | 3.2%                  |
| misfire rate (wrong command fired)           | 10.8%                | 12.9%                 |
| accuracy 95% CI                              | [75-86%]             | [73-84%]              |
| false accept 95% CI                          | [57-93%] (13/16)     | [57-93%]              |

Responses: 95.5% of trials fired a command; no response: 9; extra fires: 0; wake detect rate: -

**93-class output:** 193 command line(s) named one of the 93 phrases. Exact wording (the chosen phrase must be the spoken one): accuracy 26.7% [21-33%], balanced 27.3%, F1 13.6%, F2 19.2%. The 93-command column above uses the same rule as for every student (intent + slot right), so it stays comparable.

## Overall vs real vs synthetic voices

Each group is scored on its own. '-' = the group has no clips of that kind. The holdout's 10 out-of-scope clips are all real recordings (none are synthetic), so there is no false accept rate for synthetic voices.

| metric                         | overall        | real voice     | synthetic voice |
|--------------------------------|----------------|----------------|-----------------|
| clips (with wake word)         | 202            | 96             | 106             |
| **19 intents** accuracy        | 80.7% [75-86%] | 81.2% [72-88%] | 80.2% [72-87%]  |
| balanced accuracy              | 80.9%          | 83.4%          | 78.5%           |
| F1 (macro)                     | 79.2%          | 79.9%          | 77.4%           |
| F2 (macro)                     | 79.7%          | 81.2%          | 77.4%           |
| false accept rate              | 81.2% (13/16)  | 70.0% (7/10)   | 100.0% (6/6)    |
| false reject rate              | 3.2%           | 1.2%           | 5.0%            |
| misfire rate                   | 10.8%          | 11.6%          | 10.0%           |
| **93 commands** accuracy       | 78.7%          | 80.2%          | 77.4%           |
| balanced accuracy              | 83.2%          | 85.4%          | 83.0%           |
| F1 (macro)                     | 82.8%          | 81.4%          | 79.5%           |
| F2 (macro)                     | 82.0%          | 83.0%          | 80.8%           |
| misfire rate                   | 12.9%          | 12.8%          | 13.0%           |
| slot exact (intent right)      | 95.8% (n=96)   | 97.6% (n=42)   | 94.4% (n=54)    |
| latency p50 / p95              | 0.14 / 0.26 s  | 0.19 / 0.30 s  | 0.08 / 0.15 s   |
| false wake rate (no wake word) | -              | -              | -               |

## Slot values (slotted intents, intent right)

abs error = Manhattan (L1) distance in the slot's unit (alarm: minutes, circular over 24 h); rel error = abs error / spread of the 3 schema values; phonetic / char distance = normalised edit distance (0 same, 1 completely different) of simplified-Metaphone keys / spelled-out text.

| intent          | n  | exact  | mean abs error | mean rel error | phonetic dist | char dist |
|-----------------|----|--------|----------------|----------------|---------------|-----------|
| ALARM           | 18 | 100.0% | 0.0 min        | 0.000          | 0.000         | 0.000     |
| BRIGHTNESS      | 17 | 82.4%  | 9.4 %          | 0.118          | 0.072         | 0.079     |
| COLOR           | 13 | 92.3%  | -              | -              | 0.077         | 0.077     |
| CREATE_REMINDER | 15 | 100.0% | -              | -              | 0.000         | 0.000     |
| TEMPERATURE     | 17 | 100.0% | 0.0 deg        | 0.000          | 0.000         | 0.000     |
| TIMER           | 16 | 100.0% | 0.0 s          | 0.000          | 0.000         | 0.000     |
| ALL             | 96 | 95.8%  | -              | 0.029          | 0.023         | 0.024     |

## Raspberry Pi

- **?** (None), None cores  up to None MHz, RAM None MB, None, kernel None, Python None
- packages: -

| metric                                      | mean / p95 / max         |
|---------------------------------------------|--------------------------|
| response latency (command end -> Pi output) | 0.142 / 0.257 / 0.380 s  |
| latency p50 / p99                           | 0.141 / 0.323 s          |
| inference time (Pi-reported)                | 142.5 / 257.1 / 379.5 ms |
| real-time factor (infer / audio window)     | 0.047 / 0.067 / 0.094    |
| CPU temperature                             | -                        |
| CPU use, whole Pi                           | -                        |
| CPU use, your runtime process               | -                        |
| RAM (RSS), your runtime process             | -                        |
| RAM used, whole Pi                          | -                        |
| CPU clock                                   | -                        |
| load average (1 min)                        | -                        |
| runtime CPU-seconds per second of speech    | -                        |
| runtime CPU share of wall time              | -                        |
| throttling flags seen                       | none                     |
| test wall time                              | 0.0 min                  |

## Most frequent confusions

**intent level:** REJECT -> VOLUME_UP (4); COLOR -> REJECT (3); REJECT -> COLOR (3); LIGHT_ON -> LIGHT_OFF (2); CREATE_REMINDER -> TIME (2); REJECT -> LIGHT_OFF (2); PLAY_MUSIC -> NEXT (1); PLAY_MUSIC -> REJECT (1); LIGHT_OFF -> ALARM (1); STOP -> PLAY_MUSIC (1)

**command level:** REJECT -> Volume up (4); REJECT -> Lights out (2); Play music -> Next song (1); Play some music -> REJECT (1); Lights on -> Lights out (1); Turn on the lights -> Lights out (1); Shut off the lights -> Alarm 9:00 PM (1); Stop -> Play music (1); End playback -> Time (1); Increase the volume -> Volume down (1)

## Per-intent scores

| class           | n  | precision | recall | F1     | F2     |
|-----------------|----|-----------|--------|--------|--------|
| ALARM           | 18 | 94.7%     | 100.0% | 97.3%  | 98.9%  |
| BRIGHTNESS      | 18 | 94.4%     | 94.4%  | 94.4%  | 94.4%  |
| CALL            | 6  | 100.0%    | 50.0%  | 66.7%  | 55.6%  |
| COLOR           | 18 | 72.2%     | 72.2%  | 72.2%  | 72.2%  |
| CREATE_REMINDER | 18 | 100.0%    | 83.3%  | 90.9%  | 86.2%  |
| LIGHT_OFF       | 6  | 41.7%     | 83.3%  | 55.6%  | 69.4%  |
| LIGHT_ON        | 6  | 80.0%     | 66.7%  | 72.7%  | 69.0%  |
| LIST_REMINDERS  | 6  | 100.0%    | 66.7%  | 80.0%  | 71.4%  |
| MESSAGE         | 6  | 100.0%    | 100.0% | 100.0% | 100.0% |
| NEXT            | 6  | 85.7%     | 100.0% | 92.3%  | 96.8%  |
| PAUSE           | 6  | 100.0%    | 100.0% | 100.0% | 100.0% |
| PLAY_MUSIC      | 6  | 66.7%     | 66.7%  | 66.7%  | 66.7%  |
| REJECT          | 16 | 33.3%     | 18.8%  | 24.0%  | 20.5%  |
| STOP            | 6  | 100.0%    | 66.7%  | 80.0%  | 71.4%  |
| TEMPERATURE     | 18 | 100.0%    | 94.4%  | 97.1%  | 95.5%  |
| TIME            | 6  | 42.9%     | 100.0% | 60.0%  | 78.9%  |
| TIMER           | 18 | 94.1%     | 88.9%  | 91.4%  | 89.9%  |
| VOLUME_DOWN     | 6  | 75.0%     | 100.0% | 85.7%  | 93.8%  |
| VOLUME_UP       | 6  | 50.0%     | 66.7%  | 57.1%  | 62.5%  |
| WEATHER         | 6  | 100.0%    | 100.0% | 100.0% | 100.0% |

Scoring notes: REJECT = out-of-scope truth, or the Pi answered out-of-scope / did not respond. Command level: a prediction matches a variation when intent and slot are right (the Pi does not predict the wording); wrong predictions count against the first variation of their (intent, slot). Macro scores average over classes present in the holdout. False accept rate rests on only the out-of-scope clips in the holdout, so read its confidence interval. False wake rate: in-scope commands played WITHOUT the wake word (as many as the out-of-scope clips); any command the Pi fires for them is a false wake. These trials are not part of the 19/93 scores.
