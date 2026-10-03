> ME2 v8 - causal Conformer + CTC (+1000 negatives, reject-empty)

# VCM benchmark - v8neg - offline

Wake word: **hey rhasspy (offline: not used)** - trials: 202 with the wake word + 0 without - shuffle seed: n/a - connection: offline - holdout: airimonda/ai231-me2-voice-commands split=holdout (202 clips)

## At a glance

|                                   | overall       | real voice   | synthetic voice |
|-----------------------------------|---------------|--------------|-----------------|
| intent accuracy (19)              | 85.1%         | 75.0%        | 94.3%           |
| command accuracy (93)             | 80.2%         | 69.8%        | 89.6%           |
| false accept (out of scope fired) | 62.5% (10/16) | 60.0% (6/10) | 66.7% (4/6)     |
| false reject (command ignored)    | 1.6%          | 3.5%         | 0.0%            |
| false wake (no wake word, fired)  | -             | -            | -               |
| slot exact                        | 89.6%         | 85.7%        | 91.8%           |
| latency p95                       | 0.07 s        | 0.08 s       | 0.04 s          |

**Pi:** real-time factor 0.017 (p95 0.023), inference 48 ms

# Detailed metrics

## Classification

| metric                                       | 19 intents (+reject) | 93 commands (+reject) |
|----------------------------------------------|----------------------|-----------------------|
| accuracy                                     | 85.1%                | 80.2%                 |
| balanced accuracy                            | 86.9%                | 83.4%                 |
| precision (macro)                            | 84.4%                | 87.6%                 |
| recall (macro)                               | 86.9%                | 83.4%                 |
| F1 (macro)                                   | 83.9%                | 83.1%                 |
| F2 (macro)                                   | 85.1%                | 82.6%                 |
| false accept rate (OOS fired)                | 62.5%                | 62.5%                 |
| false reject rate (in-scope silent/rejected) | 1.6%                 | 1.6%                  |
| misfire rate (wrong command fired)           | 9.1%                 | 14.5%                 |
| accuracy 95% CI                              | [80-89%]             | [74-85%]              |
| false accept 95% CI                          | [39-82%] (10/16)     | [39-82%]              |

Responses: 95.5% of trials fired a command; no response: 9; extra fires: 0; wake detect rate: -

**93-class output:** 187 command line(s) named one of the 93 phrases. Exact wording (the chosen phrase must be the spoken one): accuracy 28.2% [22-35%], balanced 27.5%, F1 14.6%, F2 19.9%. The 93-command column above uses the same rule as for every student (intent + slot right), so it stays comparable.

## Overall vs real vs synthetic voices

Each group is scored on its own. '-' = the group has no clips of that kind. The holdout's 10 out-of-scope clips are all real recordings (none are synthetic), so there is no false accept rate for synthetic voices.

| metric                         | overall        | real voice     | synthetic voice |
|--------------------------------|----------------|----------------|-----------------|
| clips (with wake word)         | 202            | 96             | 106             |
| **19 intents** accuracy        | 85.1% [80-89%] | 75.0% [65-83%] | 94.3% [88-97%]  |
| balanced accuracy              | 86.9%          | 78.0%          | 93.3%           |
| F1 (macro)                     | 83.9%          | 73.7%          | 91.6%           |
| F2 (macro)                     | 85.1%          | 75.3%          | 92.3%           |
| false accept rate              | 62.5% (10/16)  | 60.0% (6/10)   | 66.7% (4/6)     |
| false reject rate              | 1.6%           | 3.5%           | 0.0%            |
| misfire rate                   | 9.1%           | 17.4%          | 2.0%            |
| **93 commands** accuracy       | 80.2%          | 69.8%          | 89.6%           |
| balanced accuracy              | 83.4%          | 72.9%          | 92.4%           |
| F1 (macro)                     | 83.1%          | 68.7%          | 90.4%           |
| F2 (macro)                     | 82.6%          | 70.5%          | 91.2%           |
| misfire rate                   | 14.5%          | 23.3%          | 7.0%            |
| slot exact (intent right)      | 89.6% (n=96)   | 85.7% (n=35)   | 91.8% (n=61)    |
| latency p50 / p95              | 0.04 / 0.07 s  | 0.07 / 0.08 s  | 0.03 / 0.04 s   |
| false wake rate (no wake word) | -              | -              | -               |

## Slot values (slotted intents, intent right)

abs error = Manhattan (L1) distance in the slot's unit (alarm: minutes, circular over 24 h); rel error = abs error / spread of the 3 schema values; phonetic / char distance = normalised edit distance (0 same, 1 completely different) of simplified-Metaphone keys / spelled-out text.

| intent          | n  | exact  | mean abs error | mean rel error | phonetic dist | char dist |
|-----------------|----|--------|----------------|----------------|---------------|-----------|
| ALARM           | 17 | 58.8%  | 10.9 min       | 0.012          | 0.382         | 0.382     |
| BRIGHTNESS      | 15 | 100.0% | 0.0 %          | 0.000          | 0.000         | 0.000     |
| COLOR           | 13 | 84.6%  | -              | -              | 0.154         | 0.123     |
| CREATE_REMINDER | 15 | 100.0% | -              | -              | 0.000         | 0.000     |
| TEMPERATURE     | 18 | 94.4%  | 0.2 deg        | 0.028          | 0.028         | 0.028     |
| TIMER           | 18 | 100.0% | 0.0 s          | 0.000          | 0.000         | 0.000     |
| ALL             | 96 | 89.6%  | -              | 0.010          | 0.094         | 0.090     |

## Raspberry Pi

- **?** (None), None cores  up to None MHz, RAM None MB, None, kernel None, Python None
- packages: -

| metric                                      | mean / p95 / max        |
|---------------------------------------------|-------------------------|
| response latency (command end -> Pi output) | 0.048 / 0.073 / 0.082 s |
| latency p50 / p99                           | 0.042 / 0.078 s         |
| inference time (Pi-reported)                | 48.1 / 73.1 / 82.0 ms   |
| real-time factor (infer / audio window)     | 0.017 / 0.023 / 0.035   |
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

**intent level:** COLOR -> LIGHT_OFF (3); REJECT -> TIME (3); REJECT -> PAUSE (3); LIGHT_OFF -> LIGHT_ON (2); BRIGHTNESS -> CREATE_REMINDER (2); COLOR -> PAUSE (2); CREATE_REMINDER -> REJECT (2); REJECT -> MESSAGE (2); PLAY_MUSIC -> NEXT (1); PLAY_MUSIC -> PAUSE (1)

**command level:** REJECT -> Time (3); REJECT -> Pause (3); Wake me up at 6:00 AM -> ALARM:? (2); Wake me up at 8:00 AM -> ALARM:? (2); Wake me up at 9:00 PM -> ALARM:? (2); REJECT -> Message (2); Play music -> Next song (1); Play some music -> Pause (1); What time is it? -> Pause (1); Lights out -> Lights on (1)

## Per-intent scores

| class           | n  | precision | recall | F1     | F2     |
|-----------------|----|-----------|--------|--------|--------|
| ALARM           | 18 | 100.0%    | 94.4%  | 97.1%  | 95.5%  |
| BRIGHTNESS      | 18 | 100.0%    | 83.3%  | 90.9%  | 86.2%  |
| CALL            | 6  | 83.3%     | 83.3%  | 83.3%  | 83.3%  |
| COLOR           | 18 | 100.0%    | 72.2%  | 83.9%  | 76.5%  |
| CREATE_REMINDER | 18 | 88.2%     | 83.3%  | 85.7%  | 84.3%  |
| LIGHT_OFF       | 6  | 57.1%     | 66.7%  | 61.5%  | 64.5%  |
| LIGHT_ON        | 6  | 60.0%     | 100.0% | 75.0%  | 88.2%  |
| LIST_REMINDERS  | 6  | 85.7%     | 100.0% | 92.3%  | 96.8%  |
| MESSAGE         | 6  | 75.0%     | 100.0% | 85.7%  | 93.8%  |
| NEXT            | 6  | 85.7%     | 100.0% | 92.3%  | 96.8%  |
| PAUSE           | 6  | 40.0%     | 100.0% | 57.1%  | 76.9%  |
| PLAY_MUSIC      | 6  | 100.0%    | 66.7%  | 80.0%  | 71.4%  |
| REJECT          | 16 | 66.7%     | 37.5%  | 48.0%  | 41.1%  |
| STOP            | 6  | 83.3%     | 83.3%  | 83.3%  | 83.3%  |
| TEMPERATURE     | 18 | 100.0%    | 100.0% | 100.0% | 100.0% |
| TIME            | 6  | 62.5%     | 83.3%  | 71.4%  | 78.1%  |
| TIMER           | 18 | 100.0%    | 100.0% | 100.0% | 100.0% |
| VOLUME_DOWN     | 6  | 100.0%    | 83.3%  | 90.9%  | 86.2%  |
| VOLUME_UP       | 6  | 100.0%    | 100.0% | 100.0% | 100.0% |
| WEATHER         | 6  | 100.0%    | 100.0% | 100.0% | 100.0% |

Scoring notes: REJECT = out-of-scope truth, or the Pi answered out-of-scope / did not respond. Command level: a prediction matches a variation when intent and slot are right (the Pi does not predict the wording); wrong predictions count against the first variation of their (intent, slot). Macro scores average over classes present in the holdout. False accept rate rests on only the out-of-scope clips in the holdout, so read its confidence interval. False wake rate: in-scope commands played WITHOUT the wake word (as many as the out-of-scope clips); any command the Pi fires for them is a false wake. These trials are not part of the 19/93 scores.
