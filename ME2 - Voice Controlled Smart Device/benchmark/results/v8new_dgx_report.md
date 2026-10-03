> ME2 v8 - causal Conformer + CTC (736-word vocab, content-based reject)

# VCM benchmark - v8new - offline

Wake word: **hey rhasspy (offline: not used)** - trials: 202 with the wake word + 0 without - shuffle seed: n/a - connection: offline - holdout: airimonda/ai231-me2-voice-commands split=holdout (202 clips)

## At a glance

|                                   | overall       | real voice   | synthetic voice |
|-----------------------------------|---------------|--------------|-----------------|
| intent accuracy (19)              | 85.1%         | 77.1%        | 92.5%           |
| command accuracy (93)             | 81.7%         | 72.9%        | 89.6%           |
| false accept (out of scope fired) | 87.5% (14/16) | 90.0% (9/10) | 83.3% (5/6)     |
| false reject (command ignored)    | 2.2%          | 2.3%         | 2.0%            |
| false wake (no wake word, fired)  | -             | -            | -               |
| slot exact                        | 93.1%         | 90.0%        | 95.1%           |
| latency p95                       | 0.07 s        | 0.07 s       | 0.04 s          |

**Pi:** real-time factor 0.017 (p95 0.024), inference 50 ms

# Detailed metrics

## Classification

| metric                                       | 19 intents (+reject) | 93 commands (+reject) |
|----------------------------------------------|----------------------|-----------------------|
| accuracy                                     | 85.1%                | 81.7%                 |
| balanced accuracy                            | 86.2%                | 86.8%                 |
| precision (macro)                            | 83.2%                | 89.7%                 |
| recall (macro)                               | 86.2%                | 86.8%                 |
| F1 (macro)                                   | 83.1%                | 86.4%                 |
| F2 (macro)                                   | 84.5%                | 86.1%                 |
| false accept rate (OOS fired)                | 87.5%                | 87.5%                 |
| false reject rate (in-scope silent/rejected) | 2.2%                 | 2.2%                  |
| misfire rate (wrong command fired)           | 6.5%                 | 10.2%                 |
| accuracy 95% CI                              | [80-89%]             | [76-86%]              |
| false accept 95% CI                          | [64-97%] (14/16)     | [64-97%]              |

Responses: 97.0% of trials fired a command; no response: 6; extra fires: 0; wake detect rate: -

**93-class output:** 190 command line(s) named one of the 93 phrases. Exact wording (the chosen phrase must be the spoken one): accuracy 26.2% [21-33%], balanced 27.3%, F1 14.0%, F2 19.5%. The 93-command column above uses the same rule as for every student (intent + slot right), so it stays comparable.

## Overall vs real vs synthetic voices

Each group is scored on its own. '-' = the group has no clips of that kind. The holdout's 10 out-of-scope clips are all real recordings (none are synthetic), so there is no false accept rate for synthetic voices.

| metric                         | overall        | real voice     | synthetic voice |
|--------------------------------|----------------|----------------|-----------------|
| clips (with wake word)         | 202            | 96             | 106             |
| **19 intents** accuracy        | 85.1% [80-89%] | 77.1% [68-84%] | 92.5% [86-96%]  |
| balanced accuracy              | 86.2%          | 79.9%          | 90.8%           |
| F1 (macro)                     | 83.1%          | 75.2%          | 89.3%           |
| F2 (macro)                     | 84.5%          | 77.1%          | 90.1%           |
| false accept rate              | 87.5% (14/16)  | 90.0% (9/10)   | 83.3% (5/6)     |
| false reject rate              | 2.2%           | 2.3%           | 2.0%            |
| misfire rate                   | 6.5%           | 12.8%          | 1.0%            |
| **93 commands** accuracy       | 81.7%          | 72.9%          | 89.6%           |
| balanced accuracy              | 86.8%          | 79.4%          | 92.7%           |
| F1 (macro)                     | 86.4%          | 75.9%          | 91.9%           |
| F2 (macro)                     | 86.1%          | 77.3%          | 92.3%           |
| misfire rate                   | 10.2%          | 17.4%          | 4.0%            |
| slot exact (intent right)      | 93.1% (n=101)  | 90.0% (n=40)   | 95.1% (n=61)    |
| latency p50 / p95              | 0.04 / 0.07 s  | 0.07 / 0.07 s  | 0.03 / 0.04 s   |
| false wake rate (no wake word) | -              | -              | -               |

## Slot values (slotted intents, intent right)

abs error = Manhattan (L1) distance in the slot's unit (alarm: minutes, circular over 24 h); rel error = abs error / spread of the 3 schema values; phonetic / char distance = normalised edit distance (0 same, 1 completely different) of simplified-Metaphone keys / spelled-out text.

| intent          | n   | exact  | mean abs error | mean rel error | phonetic dist | char dist |
|-----------------|-----|--------|----------------|----------------|---------------|-----------|
| ALARM           | 18  | 66.7%  | 0.0 min        | 0.000          | 0.333         | 0.333     |
| BRIGHTNESS      | 16  | 100.0% | 0.0 %          | 0.000          | 0.000         | 0.000     |
| COLOR           | 14  | 100.0% | -              | -              | 0.000         | 0.000     |
| CREATE_REMINDER | 17  | 100.0% | -              | -              | 0.000         | 0.000     |
| TEMPERATURE     | 18  | 94.4%  | 0.2 deg        | 0.028          | 0.028         | 0.028     |
| TIMER           | 18  | 100.0% | 0.0 s          | 0.000          | 0.000         | 0.000     |
| ALL             | 101 | 93.1%  | -              | 0.008          | 0.064         | 0.064     |

## Raspberry Pi

- **?** (None), None cores  up to None MHz, RAM None MB, None, kernel None, Python None
- packages: -

| metric                                      | mean / p95 / max        |
|---------------------------------------------|-------------------------|
| response latency (command end -> Pi output) | 0.050 / 0.069 / 0.689 s |
| latency p50 / p99                           | 0.041 / 0.075 s         |
| inference time (Pi-reported)                | 49.8 / 69.0 / 689.3 ms  |
| real-time factor (infer / audio window)     | 0.017 / 0.024 / 0.138   |
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

**intent level:** REJECT -> MESSAGE (3); REJECT -> PAUSE (3); REJECT -> TIME (3); LIGHT_OFF -> LIGHT_ON (2); COLOR -> LIGHT_OFF (2); REJECT -> BRIGHTNESS (2); PLAY_MUSIC -> NEXT (1); PLAY_MUSIC -> PAUSE (1); WEATHER -> REJECT (1); TIME -> PAUSE (1)

**command level:** REJECT -> Message (3); REJECT -> Pause (3); REJECT -> Time (3); Wake me up at 6:00 AM -> ALARM:? (2); Wake me up at 8:00 AM -> ALARM:? (2); Wake me up at 9:00 PM -> ALARM:? (2); REJECT -> Brightness 20 percent (2); Play music -> Next song (1); Play some music -> Pause (1); Weather -> REJECT (1)

## Per-intent scores

| class           | n  | precision | recall | F1     | F2     |
|-----------------|----|-----------|--------|--------|--------|
| ALARM           | 18 | 100.0%    | 100.0% | 100.0% | 100.0% |
| BRIGHTNESS      | 18 | 88.9%     | 88.9%  | 88.9%  | 88.9%  |
| CALL            | 6  | 100.0%    | 83.3%  | 90.9%  | 86.2%  |
| COLOR           | 18 | 100.0%    | 77.8%  | 87.5%  | 81.4%  |
| CREATE_REMINDER | 18 | 100.0%    | 94.4%  | 97.1%  | 95.5%  |
| LIGHT_OFF       | 6  | 66.7%     | 66.7%  | 66.7%  | 66.7%  |
| LIGHT_ON        | 6  | 60.0%     | 100.0% | 75.0%  | 88.2%  |
| LIST_REMINDERS  | 6  | 100.0%    | 100.0% | 100.0% | 100.0% |
| MESSAGE         | 6  | 66.7%     | 100.0% | 80.0%  | 90.9%  |
| NEXT            | 6  | 75.0%     | 100.0% | 85.7%  | 93.8%  |
| PAUSE           | 6  | 46.2%     | 100.0% | 63.2%  | 81.1%  |
| PLAY_MUSIC      | 6  | 100.0%    | 66.7%  | 80.0%  | 71.4%  |
| REJECT          | 16 | 33.3%     | 12.5%  | 18.2%  | 14.3%  |
| STOP            | 6  | 100.0%    | 83.3%  | 90.9%  | 86.2%  |
| TEMPERATURE     | 18 | 100.0%    | 100.0% | 100.0% | 100.0% |
| TIME            | 6  | 50.0%     | 66.7%  | 57.1%  | 62.5%  |
| TIMER           | 18 | 94.7%     | 100.0% | 97.3%  | 98.9%  |
| VOLUME_DOWN     | 6  | 100.0%    | 100.0% | 100.0% | 100.0% |
| VOLUME_UP       | 6  | 100.0%    | 100.0% | 100.0% | 100.0% |
| WEATHER         | 6  | 83.3%     | 83.3%  | 83.3%  | 83.3%  |

Scoring notes: REJECT = out-of-scope truth, or the Pi answered out-of-scope / did not respond. Command level: a prediction matches a variation when intent and slot are right (the Pi does not predict the wording); wrong predictions count against the first variation of their (intent, slot). Macro scores average over classes present in the holdout. False accept rate rests on only the out-of-scope clips in the holdout, so read its confidence interval. False wake rate: in-scope commands played WITHOUT the wake word (as many as the out-of-scope clips); any command the Pi fires for them is a false wake. These trials are not part of the 19/93 scores.
