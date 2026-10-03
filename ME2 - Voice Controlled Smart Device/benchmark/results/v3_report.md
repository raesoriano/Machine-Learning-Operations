> ME2 v3 - PocketSphinx ensemble (custom + stock AM, stage-2 classifier)

# VCM benchmark - v3 - offline

Wake word: **hey rhasspy (offline: not used)** - trials: 202 with the wake word + 0 without - shuffle seed: n/a - connection: offline - holdout: airimonda/ai231-me2-voice-commands split=holdout (202 clips)

## At a glance

|                                   | overall       | real voice   | synthetic voice |
|-----------------------------------|---------------|--------------|-----------------|
| intent accuracy (19)              | 82.7%         | 80.2%        | 84.9%           |
| command accuracy (93)             | 80.7%         | 79.2%        | 82.1%           |
| false accept (out of scope fired) | 87.5% (14/16) | 80.0% (8/10) | 100.0% (6/6)    |
| false reject (command ignored)    | 2.7%          | 1.2%         | 4.0%            |
| false wake (no wake word, fired)  | -             | -            | -               |
| slot exact                        | 96.0%         | 97.7%        | 94.6%           |
| latency p95                       | 0.23 s        | 0.29 s       | 0.13 s          |

**Pi:** real-time factor 0.044 (p95 0.062), inference 134 ms

# Detailed metrics

## Classification

| metric                                       | 19 intents (+reject) | 93 commands (+reject) |
|----------------------------------------------|----------------------|-----------------------|
| accuracy                                     | 82.7%                | 80.7%                 |
| balanced accuracy                            | 83.1%                | 85.8%                 |
| precision (macro)                            | 80.4%                | 87.0%                 |
| recall (macro)                               | 83.1%                | 85.8%                 |
| F1 (macro)                                   | 80.1%                | 84.1%                 |
| F2 (macro)                                   | 81.6%                | 84.4%                 |
| false accept rate (OOS fired)                | 87.5%                | 87.5%                 |
| false reject rate (in-scope silent/rejected) | 2.7%                 | 2.7%                  |
| misfire rate (wrong command fired)           | 8.6%                 | 10.8%                 |
| accuracy 95% CI                              | [77-87%]             | [75-86%]              |
| false accept 95% CI                          | [64-97%] (14/16)     | [64-97%]              |

Responses: 96.5% of trials fired a command; no response: 7; extra fires: 0; wake detect rate: -

**93-class output:** 195 command line(s) named one of the 93 phrases. Exact wording (the chosen phrase must be the spoken one): accuracy 27.7% [22-34%], balanced 28.9%, F1 14.1%, F2 20.2%. The 93-command column above uses the same rule as for every student (intent + slot right), so it stays comparable.

## Overall vs real vs synthetic voices

Each group is scored on its own. '-' = the group has no clips of that kind. The holdout's 10 out-of-scope clips are all real recordings (none are synthetic), so there is no false accept rate for synthetic voices.

| metric                         | overall        | real voice     | synthetic voice |
|--------------------------------|----------------|----------------|-----------------|
| clips (with wake word)         | 202            | 96             | 106             |
| **19 intents** accuracy        | 82.7% [77-87%] | 80.2% [71-87%] | 84.9% [77-90%]  |
| balanced accuracy              | 83.1%          | 82.1%          | 84.6%           |
| F1 (macro)                     | 80.1%          | 77.7%          | 82.3%           |
| F2 (macro)                     | 81.6%          | 79.6%          | 83.2%           |
| false accept rate              | 87.5% (14/16)  | 80.0% (8/10)   | 100.0% (6/6)    |
| false reject rate              | 2.7%           | 1.2%           | 4.0%            |
| misfire rate                   | 8.6%           | 11.6%          | 6.0%            |
| **93 commands** accuracy       | 80.7%          | 79.2%          | 82.1%           |
| balanced accuracy              | 85.8%          | 85.3%          | 88.8%           |
| F1 (macro)                     | 84.1%          | 81.1%          | 85.2%           |
| F2 (macro)                     | 84.4%          | 82.9%          | 86.8%           |
| misfire rate                   | 10.8%          | 12.8%          | 9.0%            |
| slot exact (intent right)      | 96.0% (n=99)   | 97.7% (n=43)   | 94.6% (n=56)    |
| latency p50 / p95              | 0.12 / 0.23 s  | 0.18 / 0.29 s  | 0.08 / 0.13 s   |
| false wake rate (no wake word) | -              | -              | -               |

## Slot values (slotted intents, intent right)

abs error = Manhattan (L1) distance in the slot's unit (alarm: minutes, circular over 24 h); rel error = abs error / spread of the 3 schema values; phonetic / char distance = normalised edit distance (0 same, 1 completely different) of simplified-Metaphone keys / spelled-out text.

| intent          | n  | exact  | mean abs error | mean rel error | phonetic dist | char dist |
|-----------------|----|--------|----------------|----------------|---------------|-----------|
| ALARM           | 18 | 100.0% | 0.0 min        | 0.000          | 0.000         | 0.000     |
| BRIGHTNESS      | 18 | 83.3%  | 8.9 %          | 0.111          | 0.068         | 0.074     |
| COLOR           | 13 | 92.3%  | -              | -              | 0.077         | 0.062     |
| CREATE_REMINDER | 15 | 100.0% | -              | -              | 0.000         | 0.000     |
| TEMPERATURE     | 17 | 100.0% | 0.0 deg        | 0.000          | 0.000         | 0.000     |
| TIMER           | 18 | 100.0% | 0.0 s          | 0.000          | 0.000         | 0.000     |
| ALL             | 99 | 96.0%  | -              | 0.028          | 0.023         | 0.022     |

## Raspberry Pi

- **?** (None), None cores  up to None MHz, RAM None MB, None, kernel None, Python None
- packages: -

| metric                                      | mean / p95 / max         |
|---------------------------------------------|--------------------------|
| response latency (command end -> Pi output) | 0.134 / 0.230 / 0.689 s  |
| latency p50 / p99                           | 0.123 / 0.320 s          |
| inference time (Pi-reported)                | 134.3 / 229.6 / 688.7 ms |
| real-time factor (infer / audio window)     | 0.044 / 0.062 / 0.138    |
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

**intent level:** REJECT -> VOLUME_UP (4); REJECT -> COLOR (3); CALL -> REJECT (2); COLOR -> PLAY_MUSIC (2); COLOR -> REJECT (2); REJECT -> LIGHT_ON (2); PLAY_MUSIC -> NEXT (1); PLAY_MUSIC -> REJECT (1); TIME -> LIST_REMINDERS (1); LIGHT_OFF -> PLAY_MUSIC (1)

**command level:** REJECT -> Volume up (4); REJECT -> Lights on (2); Play music -> Next song (1); Play some music -> REJECT (1); Tell me the time -> Reminders (1); Lights out -> Play music (1); Stop -> Play music (1); End playback -> Time (1); Turn the volume up -> Reminders (1); Call -> Change color to Blue (1)

## Per-intent scores

| class           | n  | precision | recall | F1     | F2     |
|-----------------|----|-----------|--------|--------|--------|
| ALARM           | 18 | 100.0%    | 100.0% | 100.0% | 100.0% |
| BRIGHTNESS      | 18 | 100.0%    | 100.0% | 100.0% | 100.0% |
| CALL            | 6  | 100.0%    | 33.3%  | 50.0%  | 38.5%  |
| COLOR           | 18 | 72.2%     | 72.2%  | 72.2%  | 72.2%  |
| CREATE_REMINDER | 18 | 93.8%     | 83.3%  | 88.2%  | 85.2%  |
| LIGHT_OFF       | 6  | 83.3%     | 83.3%  | 83.3%  | 83.3%  |
| LIGHT_ON        | 6  | 75.0%     | 100.0% | 85.7%  | 93.8%  |
| LIST_REMINDERS  | 6  | 71.4%     | 83.3%  | 76.9%  | 80.6%  |
| MESSAGE         | 6  | 100.0%    | 100.0% | 100.0% | 100.0% |
| NEXT            | 6  | 85.7%     | 100.0% | 92.3%  | 96.8%  |
| PAUSE           | 6  | 100.0%    | 100.0% | 100.0% | 100.0% |
| PLAY_MUSIC      | 6  | 36.4%     | 66.7%  | 47.1%  | 57.1%  |
| REJECT          | 16 | 28.6%     | 12.5%  | 17.4%  | 14.1%  |
| STOP            | 6  | 80.0%     | 66.7%  | 72.7%  | 69.0%  |
| TEMPERATURE     | 18 | 100.0%    | 94.4%  | 97.1%  | 95.5%  |
| TIME            | 6  | 71.4%     | 83.3%  | 76.9%  | 80.6%  |
| TIMER           | 18 | 94.7%     | 100.0% | 97.3%  | 98.9%  |
| VOLUME_DOWN     | 6  | 85.7%     | 100.0% | 92.3%  | 96.8%  |
| VOLUME_UP       | 6  | 55.6%     | 83.3%  | 66.7%  | 75.8%  |
| WEATHER         | 6  | 75.0%     | 100.0% | 85.7%  | 93.8%  |

Scoring notes: REJECT = out-of-scope truth, or the Pi answered out-of-scope / did not respond. Command level: a prediction matches a variation when intent and slot are right (the Pi does not predict the wording); wrong predictions count against the first variation of their (intent, slot). Macro scores average over classes present in the holdout. False accept rate rests on only the out-of-scope clips in the holdout, so read its confidence interval. False wake rate: in-scope commands played WITHOUT the wake word (as many as the out-of-scope clips); any command the Pi fires for them is a false wake. These trials are not part of the 19/93 scores.
