> ME2 v6 - from-scratch HMM/GMM (numpy), grammar-constrained + free decode

# VCM benchmark - v6 - offline

Wake word: **hey rhasspy (offline: not used)** - trials: 202 with the wake word + 0 without - shuffle seed: n/a - connection: offline - holdout: airimonda/ai231-me2-voice-commands split=holdout (202 clips)

## At a glance

|                                   | overall        | real voice     | synthetic voice |
|-----------------------------------|----------------|----------------|-----------------|
| intent accuracy (19)              | 39.6%          | 24.0%          | 53.8%           |
| command accuracy (93)             | 26.2%          | 9.4%           | 41.5%           |
| false accept (out of scope fired) | 100.0% (16/16) | 100.0% (10/10) | 100.0% (6/6)    |
| false reject (command ignored)    | 0.0%           | 0.0%           | 0.0%            |
| false wake (no wake word, fired)  | -              | -              | -               |
| slot exact                        | 57.8%          | 39.1%          | 68.3%           |
| latency p95                       | 19.19 s        | 19.41 s        | 9.37 s          |

**Pi:** real-time factor 3.357 (p95 3.844), inference 10989 ms

# Detailed metrics

## Classification

| metric                                       | 19 intents (+reject) | 93 commands (+reject) |
|----------------------------------------------|----------------------|-----------------------|
| accuracy                                     | 39.6%                | 26.2%                 |
| balanced accuracy                            | 31.1%                | 28.2%                 |
| precision (macro)                            | 32.5%                | 37.7%                 |
| recall (macro)                               | 31.1%                | 28.2%                 |
| F1 (macro)                                   | 29.1%                | 29.7%                 |
| F2 (macro)                                   | 29.9%                | 27.8%                 |
| false accept rate (OOS fired)                | 100.0%               | 100.0%                |
| false reject rate (in-scope silent/rejected) | 0.0%                 | 0.0%                  |
| misfire rate (wrong command fired)           | 57.0%                | 71.5%                 |
| accuracy 95% CI                              | [33-46%]             | [21-33%]              |
| false accept 95% CI                          | [81-100%] (16/16)    | [81-100%]             |

Responses: 100.0% of trials fired a command; no response: 0; extra fires: 0; wake detect rate: -

**93-class output:** 195 command line(s) named one of the 93 phrases. Exact wording (the chosen phrase must be the spoken one): accuracy 6.4% [4-11%], balanced 6.9%, F1 3.5%, F2 4.6%. The 93-command column above uses the same rule as for every student (intent + slot right), so it stays comparable.

## Overall vs real vs synthetic voices

Each group is scored on its own. '-' = the group has no clips of that kind. The holdout's 10 out-of-scope clips are all real recordings (none are synthetic), so there is no false accept rate for synthetic voices.

| metric                         | overall        | real voice      | synthetic voice |
|--------------------------------|----------------|-----------------|-----------------|
| clips (with wake word)         | 202            | 96              | 106             |
| **19 intents** accuracy        | 39.6% [33-46%] | 24.0% [17-33%]  | 53.8% [44-63%]  |
| balanced accuracy              | 31.1%          | 14.4%           | 46.8%           |
| F1 (macro)                     | 29.1%          | 9.7%            | 40.2%           |
| F2 (macro)                     | 29.9%          | 11.9%           | 43.2%           |
| false accept rate              | 100.0% (16/16) | 100.0% (10/10)  | 100.0% (6/6)    |
| false reject rate              | 0.0%           | 0.0%            | 0.0%            |
| misfire rate                   | 57.0%          | 73.3%           | 43.0%           |
| **93 commands** accuracy       | 26.2%          | 9.4%            | 41.5%           |
| balanced accuracy              | 28.2%          | 10.3%           | 42.6%           |
| F1 (macro)                     | 29.7%          | 8.3%            | 37.2%           |
| F2 (macro)                     | 27.8%          | 8.7%            | 39.0%           |
| misfire rate                   | 71.5%          | 89.5%           | 56.0%           |
| slot exact (intent right)      | 57.8% (n=64)   | 39.1% (n=23)    | 68.3% (n=41)    |
| latency p50 / p95              | 8.68 / 19.19 s | 17.60 / 19.41 s | 5.37 / 9.37 s   |
| false wake rate (no wake word) | -              | -               | -               |

## Slot values (slotted intents, intent right)

abs error = Manhattan (L1) distance in the slot's unit (alarm: minutes, circular over 24 h); rel error = abs error / spread of the 3 schema values; phonetic / char distance = normalised edit distance (0 same, 1 completely different) of simplified-Metaphone keys / spelled-out text.

| intent          | n  | exact  | mean abs error | mean rel error | phonetic dist | char dist |
|-----------------|----|--------|----------------|----------------|---------------|-----------|
| ALARM           | 8  | 37.5%  | 156.0 min      | 0.173          | 0.512         | 0.516     |
| BRIGHTNESS      | 7  | 100.0% | 0.0 %          | 0.000          | 0.000         | 0.000     |
| COLOR           | 15 | 46.7%  | -              | -              | 0.489         | 0.453     |
| CREATE_REMINDER | 13 | 53.8%  | -              | -              | 0.404         | 0.441     |
| TEMPERATURE     | 15 | 60.0%  | 2.4 deg        | 0.300          | 0.154         | 0.133     |
| TIMER           | 6  | 66.7%  | 6.7 s          | 0.133          | 0.111         | 0.119     |
| ALL             | 64 | 57.8%  | -              | 0.187          | 0.307         | 0.303     |

## Raspberry Pi

- **?** (None), None cores  up to None MHz, RAM None MB, None, kernel None, Python None
- packages: -

| metric                                      | mean / p95 / max               |
|---------------------------------------------|--------------------------------|
| response latency (command end -> Pi output) | 10.989 / 19.193 / 20.191 s     |
| latency p50 / p99                           | 8.676 / 19.549 s               |
| inference time (Pi-reported)                | 10989.3 / 19192.8 / 20191.2 ms |
| real-time factor (infer / audio window)     | 3.357 / 3.844 / 4.038          |
| CPU temperature                             | -                              |
| CPU use, whole Pi                           | -                              |
| CPU use, your runtime process               | -                              |
| RAM (RSS), your runtime process             | -                              |
| RAM used, whole Pi                          | -                              |
| CPU clock                                   | -                              |
| load average (1 min)                        | -                              |
| runtime CPU-seconds per second of speech    | -                              |
| runtime CPU share of wall time              | -                              |
| throttling flags seen                       | none                           |
| test wall time                              | 0.0 min                        |

## Most frequent confusions

**intent level:** TIMER -> WEATHER (4); BRIGHTNESS -> TIMER (4); BRIGHTNESS -> WEATHER (4); REJECT -> BRIGHTNESS (4); TIMER -> CREATE_REMINDER (3); ALARM -> CREATE_REMINDER (3); REJECT -> TEMPERATURE (3); REJECT -> CREATE_REMINDER (3); PLAY_MUSIC -> TIMER (2); WEATHER -> CREATE_REMINDER (2)

**command level:** REJECT -> Temperature 26 degrees (3); Countdown for 30 seconds -> Timer 10 seconds (2); Start a timer for 1 minute -> Weather (2); Alarm 9:00 PM -> Reminders (2); Brightness level 20 percent -> Timer 10 seconds (2); REJECT -> Timer 10 seconds (2); REJECT -> Brightness 20 percent (2); Play music -> Timer 1 minute (1); Play music -> Weather (1); Start music -> Brightness 20 percent (1)

## Per-intent scores

| class           | n  | precision | recall | F1    | F2    |
|-----------------|----|-----------|--------|-------|-------|
| ALARM           | 18 | 50.0%     | 44.4%  | 47.1% | 45.5% |
| BRIGHTNESS      | 18 | 36.8%     | 38.9%  | 37.8% | 38.5% |
| CALL            | 6  | 0.0%      | 0.0%   | 0.0%  | 0.0%  |
| COLOR           | 18 | 55.6%     | 83.3%  | 66.7% | 75.8% |
| CREATE_REMINDER | 18 | 38.2%     | 72.2%  | 50.0% | 61.3% |
| LIGHT_OFF       | 6  | 40.0%     | 33.3%  | 36.4% | 34.5% |
| LIGHT_ON        | 6  | 37.5%     | 50.0%  | 42.9% | 46.9% |
| LIST_REMINDERS  | 6  | 0.0%      | 0.0%   | 0.0%  | 0.0%  |
| MESSAGE         | 6  | 50.0%     | 50.0%  | 50.0% | 50.0% |
| NEXT            | 6  | 25.0%     | 16.7%  | 20.0% | 17.9% |
| PAUSE           | 6  | 50.0%     | 16.7%  | 25.0% | 19.2% |
| PLAY_MUSIC      | 6  | 0.0%      | 0.0%   | 0.0%  | 0.0%  |
| REJECT          | 16 | 0.0%      | 0.0%   | 0.0%  | 0.0%  |
| STOP            | 6  | 100.0%    | 16.7%  | 28.6% | 20.0% |
| TEMPERATURE     | 18 | 65.2%     | 83.3%  | 73.2% | 78.9% |
| TIME            | 6  | 50.0%     | 50.0%  | 50.0% | 50.0% |
| TIMER           | 18 | 37.5%     | 33.3%  | 35.3% | 34.1% |
| VOLUME_DOWN     | 6  | 0.0%      | 0.0%   | 0.0%  | 0.0%  |
| VOLUME_UP       | 6  | 0.0%      | 0.0%   | 0.0%  | 0.0%  |
| WEATHER         | 6  | 14.3%     | 33.3%  | 20.0% | 26.3% |

Scoring notes: REJECT = out-of-scope truth, or the Pi answered out-of-scope / did not respond. Command level: a prediction matches a variation when intent and slot are right (the Pi does not predict the wording); wrong predictions count against the first variation of their (intent, slot). Macro scores average over classes present in the holdout. False accept rate rests on only the out-of-scope clips in the holdout, so read its confidence interval. False wake rate: in-scope commands played WITHOUT the wake word (as many as the out-of-scope clips); any command the Pi fires for them is a false wake. These trials are not part of the 19/93 scores.
