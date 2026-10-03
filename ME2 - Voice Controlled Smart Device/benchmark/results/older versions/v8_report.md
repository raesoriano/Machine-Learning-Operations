> ME2 v8 - causal Conformer + CTC (base, score-gap reject)

# VCM benchmark - v8 - offline

Wake word: **hey rhasspy (offline: not used)** - trials: 202 with the wake word + 0 without - shuffle seed: n/a - connection: offline - holdout: airimonda/ai231-me2-voice-commands split=holdout (202 clips)

## At a glance

|                                   | overall        | real voice     | synthetic voice |
|-----------------------------------|----------------|----------------|-----------------|
| intent accuracy (19)              | 83.2%          | 72.9%          | 92.5%           |
| command accuracy (93)             | 78.2%          | 67.7%          | 87.7%           |
| false accept (out of scope fired) | 100.0% (16/16) | 100.0% (10/10) | 100.0% (6/6)    |
| false reject (command ignored)    | 0.0%           | 0.0%           | 0.0%            |
| false wake (no wake word, fired)  | -              | -              | -               |
| slot exact                        | 89.8%          | 86.5%          | 91.8%           |
| latency p95                       | 0.07 s         | 0.07 s         | 0.04 s          |

**Pi:** real-time factor 0.022 (p95 0.027), inference 74 ms

# Detailed metrics

## Classification

| metric                                       | 19 intents (+reject) | 93 commands (+reject) |
|----------------------------------------------|----------------------|-----------------------|
| accuracy                                     | 83.2%                | 78.2%                 |
| balanced accuracy                            | 85.6%                | 84.0%                 |
| precision (macro)                            | 78.2%                | 86.3%                 |
| recall (macro)                               | 85.6%                | 84.0%                 |
| F1 (macro)                                   | 80.0%                | 82.5%                 |
| F2 (macro)                                   | 82.6%                | 82.6%                 |
| false accept rate (OOS fired)                | 100.0%               | 100.0%                |
| false reject rate (in-scope silent/rejected) | 0.0%                 | 0.0%                  |
| misfire rate (wrong command fired)           | 9.7%                 | 15.1%                 |
| accuracy 95% CI                              | [77-88%]             | [72-83%]              |
| false accept 95% CI                          | [81-100%] (16/16)    | [81-100%]             |

Responses: 100.0% of trials fired a command; no response: 0; extra fires: 0; wake detect rate: -

**93-class output:** 196 command line(s) named one of the 93 phrases. Exact wording (the chosen phrase must be the spoken one): accuracy 26.2% [21-33%], balanced 28.2%, F1 14.2%, F2 19.9%. The 93-command column above uses the same rule as for every student (intent + slot right), so it stays comparable.

## Overall vs real vs synthetic voices

Each group is scored on its own. '-' = the group has no clips of that kind. The holdout's 10 out-of-scope clips are all real recordings (none are synthetic), so there is no false accept rate for synthetic voices.

| metric                         | overall        | real voice     | synthetic voice |
|--------------------------------|----------------|----------------|-----------------|
| clips (with wake word)         | 202            | 96             | 106             |
| **19 intents** accuracy        | 83.2% [77-88%] | 72.9% [63-81%] | 92.5% [86-96%]  |
| balanced accuracy              | 85.6%          | 77.2%          | 91.7%           |
| F1 (macro)                     | 80.0%          | 70.0%          | 87.8%           |
| F2 (macro)                     | 82.6%          | 73.0%          | 89.8%           |
| false accept rate              | 100.0% (16/16) | 100.0% (10/10) | 100.0% (6/6)    |
| false reject rate              | 0.0%           | 0.0%           | 0.0%            |
| misfire rate                   | 9.7%           | 18.6%          | 2.0%            |
| **93 commands** accuracy       | 78.2%          | 67.7%          | 87.7%           |
| balanced accuracy              | 84.0%          | 74.7%          | 92.0%           |
| F1 (macro)                     | 82.5%          | 68.9%          | 89.4%           |
| F2 (macro)                     | 82.6%          | 71.4%          | 90.5%           |
| misfire rate                   | 15.1%          | 24.4%          | 7.0%            |
| slot exact (intent right)      | 89.8% (n=98)   | 86.5% (n=37)   | 91.8% (n=61)    |
| latency p50 / p95              | 0.04 / 0.07 s  | 0.07 / 0.07 s  | 0.03 / 0.04 s   |
| false wake rate (no wake word) | -              | -              | -               |

## Slot values (slotted intents, intent right)

abs error = Manhattan (L1) distance in the slot's unit (alarm: minutes, circular over 24 h); rel error = abs error / spread of the 3 schema values; phonetic / char distance = normalised edit distance (0 same, 1 completely different) of simplified-Metaphone keys / spelled-out text.

| intent          | n  | exact  | mean abs error | mean rel error | phonetic dist | char dist |
|-----------------|----|--------|----------------|----------------|---------------|-----------|
| ALARM           | 17 | 58.8%  | 10.9 min       | 0.012          | 0.382         | 0.382     |
| BRIGHTNESS      | 15 | 100.0% | 0.0 %          | 0.000          | 0.000         | 0.000     |
| COLOR           | 13 | 84.6%  | -              | -              | 0.154         | 0.123     |
| CREATE_REMINDER | 17 | 100.0% | -              | -              | 0.000         | 0.000     |
| TEMPERATURE     | 18 | 94.4%  | 0.2 deg        | 0.028          | 0.028         | 0.028     |
| TIMER           | 18 | 100.0% | 0.0 s          | 0.000          | 0.000         | 0.000     |
| ALL             | 98 | 89.8%  | -              | 0.010          | 0.092         | 0.088     |

## Raspberry Pi

- **?** (None), None cores  up to None MHz, RAM None MB, None, kernel None, Python None
- packages: -

| metric                                      | mean / p95 / max        |
|---------------------------------------------|-------------------------|
| response latency (command end -> Pi output) | 0.074 / 0.072 / 5.100 s |
| latency p50 / p99                           | 0.042 / 0.080 s         |
| inference time (Pi-reported)                | 73.6 / 71.8 / 5100.0 ms |
| real-time factor (infer / audio window)     | 0.022 / 0.027 / 1.020   |
| CPU temperature                             | -                       |
| CPU use, whole Pi                           | -                       |
| CPU use, your runtime process               | -                       |
| RAM (RSS), your runtime process             | -                       |
| RAM used, whole Pi                          | -                       |
| CPU clock                                   | -                       |
| load average (1 min)                        | -                       |
| runtime CPU-seconds per second of speech    | -                       |
| runtime CPU share of wall time              | -                       |
| throttling flags seen                       | none                    |
| test wall time                              | 0.0 min                 |

## Most frequent confusions

**intent level:** REJECT -> PAUSE (4); COLOR -> LIGHT_OFF (3); REJECT -> WEATHER (3); REJECT -> TIME (3); LIGHT_OFF -> LIGHT_ON (2); BRIGHTNESS -> CREATE_REMINDER (2); COLOR -> PAUSE (2); REJECT -> MESSAGE (2); REJECT -> STOP (2); PLAY_MUSIC -> NEXT (1)

**command level:** REJECT -> Pause (4); REJECT -> Weather (3); REJECT -> Time (3); Wake me up at 6:00 AM -> ALARM:? (2); Wake me up at 8:00 AM -> ALARM:? (2); Wake me up at 9:00 PM -> ALARM:? (2); REJECT -> Message (2); REJECT -> Stop (2); Play music -> Next song (1); Play some music -> Pause (1)

## Per-intent scores

| class           | n  | precision | recall | F1     | F2     |
|-----------------|----|-----------|--------|--------|--------|
| ALARM           | 18 | 100.0%    | 94.4%  | 97.1%  | 95.5%  |
| BRIGHTNESS      | 18 | 100.0%    | 83.3%  | 90.9%  | 86.2%  |
| CALL            | 6  | 83.3%     | 83.3%  | 83.3%  | 83.3%  |
| COLOR           | 18 | 100.0%    | 72.2%  | 83.9%  | 76.5%  |
| CREATE_REMINDER | 18 | 89.5%     | 94.4%  | 91.9%  | 93.4%  |
| LIGHT_OFF       | 6  | 57.1%     | 66.7%  | 61.5%  | 64.5%  |
| LIGHT_ON        | 6  | 60.0%     | 100.0% | 75.0%  | 88.2%  |
| LIST_REMINDERS  | 6  | 85.7%     | 100.0% | 92.3%  | 96.8%  |
| MESSAGE         | 6  | 75.0%     | 100.0% | 85.7%  | 93.8%  |
| NEXT            | 6  | 85.7%     | 100.0% | 92.3%  | 96.8%  |
| PAUSE           | 6  | 35.3%     | 100.0% | 52.2%  | 73.2%  |
| PLAY_MUSIC      | 6  | 100.0%    | 66.7%  | 80.0%  | 71.4%  |
| REJECT          | 16 | 0.0%      | 0.0%   | 0.0%   | 0.0%   |
| STOP            | 6  | 62.5%     | 83.3%  | 71.4%  | 78.1%  |
| TEMPERATURE     | 18 | 100.0%    | 100.0% | 100.0% | 100.0% |
| TIME            | 6  | 62.5%     | 83.3%  | 71.4%  | 78.1%  |
| TIMER           | 18 | 100.0%    | 100.0% | 100.0% | 100.0% |
| VOLUME_DOWN     | 6  | 100.0%    | 83.3%  | 90.9%  | 86.2%  |
| VOLUME_UP       | 6  | 100.0%    | 100.0% | 100.0% | 100.0% |
| WEATHER         | 6  | 66.7%     | 100.0% | 80.0%  | 90.9%  |

Scoring notes: REJECT = out-of-scope truth, or the Pi answered out-of-scope / did not respond. Command level: a prediction matches a variation when intent and slot are right (the Pi does not predict the wording); wrong predictions count against the first variation of their (intent, slot). Macro scores average over classes present in the holdout. False accept rate rests on only the out-of-scope clips in the holdout, so read its confidence interval. False wake rate: in-scope commands played WITHOUT the wake word (as many as the out-of-scope clips); any command the Pi fires for them is a false wake. These trials are not part of the 19/93 scores.
