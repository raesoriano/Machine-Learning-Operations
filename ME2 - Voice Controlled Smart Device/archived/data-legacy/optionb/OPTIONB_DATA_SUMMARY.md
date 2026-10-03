# Option B Dataset Summary

- Original dataset: **18,600 WAV files**.
- After filtering: **17,924 active WAV files** (8,962 clean and 8,962 noisy).
- Archived in `FLAGGED/`: **730 WAV files**, including 54 older recordings whose paths are active again.
- Current excluded audio paths: **676**.
- Detailed folder counts: see [README.md](README.md).

Latest publication: **2026-09-22**. The music-command phrases and QA rules are documented in [README.md](README.md). Personal speakers and private wake/exit recordings are not published.

The tables below describe the original dataset before filtering. Active counts vary by folder and speaker.

## Original dataset overview

| Item                        | Summary                                |
| --------------------------- | -------------------------------------- |
| Server                      | DGX2                                   |
| Dataset path                |                                        |
| Total intents               | 19                                     |
| Data folders                | 31, plus a separate `FLAGGED` folder    |
| Fixed intents               | 13 folders                             |
| Variable intents            | 6 intents Ãƒâ€” 3 slot values = 18 folders |
| Phrase variations           | 3 per intent/slot combination          |
| Acoustic conditions         | `clean` and `noisy`                    |
| WAV files per data folder   | 600                                    |
| Clean WAV files             | 9,300                                  |
| Noisy WAV files             | 9,300                                  |
| **Total WAV files**         | **18,600**                             |
| Total speakers              | 100                                    |
| Base utterances per speaker | 93                                     |
| WAV files per speaker       | 186                                    |

Originally, each folder contained 100 speakers Ãƒâ€” 3 phrase variations Ãƒâ€” 2 acoustic conditions = 600 files.

## Speaker groups

| Group                         | Speaker IDs                    | Speakers | WAV files  |
| ----------------------------- | ------------------------------ | -------: | ---------: |
| Foreign (LibriSpeech)         | `s1Ã¢â‚¬â€œs67`, `s81Ã¢â‚¬â€œs88`, `s91Ã¢â‚¬â€œs99` |       84 |     15,624 |
| Filipino-English (SilencioPH) | `s68Ã¢â‚¬â€œs80`, `s89Ã¢â‚¬â€œs90`, `s100`   |       16 |      2,976 |
| **Total**                     | `s1Ã¢â‚¬â€œs100`                      |  **100** | **18,600** |

Filipino-English audio is distributed across the same 31 data folders. Identify it using the speaker IDs in the filenames.

## Speaker split assignments

| Metadata split | Foreign speaker IDs | Filipino speaker IDs | Foreign | Filipino | Total speakers |
| -------------- | ------------------- | -------------------- | ------: | -------: | -------------: |
| Train          | `s1Ã¢â‚¬â€œs67`            | `s68Ã¢â‚¬â€œs80`            |      67 |       13 |             80 |
| Validation     | `s81Ã¢â‚¬â€œs88`           | `s89Ã¢â‚¬â€œs90`            |       8 |        2 |             10 |
| Test           | `s91Ã¢â‚¬â€œs99`           | `s100`               |       9 |        1 |             10 |

Filipino speakers appear in all three metadata splits. Files are organized by intent/slot folders, not physical train/validation/test directories. These assignments come from speaker metadata; whether the training loader follows them has not been checked.

## Data folders

Before filtering, all folders listed below contained 600 WAV files each.

| Group         | Folder names                                                                                                                                       | Group WAV total |
| ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | --------------: |
| Fixed intents | `PLAY_MUSIC`, `WEATHER`, `TIME`, `LIGHT_ON`, `LIGHT_OFF`, `PAUSE`, `STOP`, `NEXT`, `VOLUME_UP`, `VOLUME_DOWN`, `CALL`, `MESSAGE`, `LIST_REMINDERS` |           7,800 |
| Alarm         | `ALARM_6_00AM`, `ALARM_8_00AM`, `ALARM_9_00PM`                                                                                                     |           1,800 |
| Timer         | `TIMER_10s`, `TIMER_30s`, `TIMER_1m`                                                                                                               |           1,800 |
| Temperature   | `TEMPERATURE_18`, `TEMPERATURE_22`, `TEMPERATURE_26`                                                                                               |           1,800 |
| Brightness    | `BRIGHTNESS_20`, `BRIGHTNESS_60`, `BRIGHTNESS_100`                                                                                                 |           1,800 |
| Color         | `COLOR_RED`, `COLOR_BLUE`, `COLOR_GREEN`                                                                                                           |           1,800 |
| Reminder      | `CREATE_REMINDER_DRINK_WATER`, `CREATE_REMINDER_STUDY`, `CREATE_REMINDER_EXERCISE`                                                                                     |           1,800 |

## Phrase variations (music commands updated 2026-09-21)

| Intent / group    | v1                    | v2                                  | v3                                 |
| ----------------- | --------------------- | ----------------------------------- | ---------------------------------- |
| `PLAY_MUSIC` | Play music | Start music | Play some music |
| `WEATHER`         | Weather               | What's the weather?                 | Tell me the weather                |
| `TIME`            | Time                  | What time is it?                    | Tell me the time                   |
| `LIGHT_ON`        | Lights on             | Power on the lights                 | Turn on the lights                 |
| `LIGHT_OFF`       | Lights off            | Kill the lights                     | Turn off the lights                |
| `PAUSE` | Pause | Pause audio | Pause song |
| `STOP` | Stop | Stop playing | Stop the music |
| `NEXT` | Next song | Skip song | Play next song |
| `VOLUME_UP`       | Volume up             | Increase the volume                 | Turn the volume up                 |
| `VOLUME_DOWN`     | Volume down           | Lower the volume                 | Turn the volume down               |
| `CALL`            | Call                  | Make a call                         | Make a phone call                  |
| `MESSAGE`         | Message               | Send a message                      | Send my message                    |
| `LIST_REMINDERS`  | Reminders             | Show my reminders                   | List my reminders                  |
| `TIMER`           | Timer {duration}      | Countdown for {duration}            | Start a timer for {duration}       |
| `ALARM`           | Alarm {time}          | Wake me up at {time}                | Set an alarm for {time}            |
| `TEMPERATURE`     | Temperature {degrees} | Change the temperature to {degrees} | Set the temperature to {degrees}   |
| `BRIGHTNESS`      | Brightness {percent}  | Set the brightness to {percent}     | Change the brightness to {percent} |
| `COLOR`           | Color {color}         | Change the lights to {color}        | Set the lights to {color}          |
| `CREATE_REMINDER` | Reminder {task}       | Remind me to {task}                 | Create a reminder to {task}        |

### Slot values

| Intent            | Slot         | Values                              |
| ----------------- | ------------ | ----------------------------------- |
| `TIMER`           | `{duration}` | 10 seconds; 30 seconds; 1 minute    |
| `ALARM`           | `{time}`     | 6 AM; 8 AM; 9 PM                    |
| `TEMPERATURE`     | `{degrees}`  | 18 degrees; 22 degrees; 26 degrees  |
| `BRIGHTNESS`      | `{percent}`  | 20 percent; 60 percent; 100 percent |
| `COLOR`           | `{color}`    | red; blue; green                    |
| `CREATE_REMINDER` | `{task}`     | drink water; study; exercise       |

## Filename convention

`<FOLDER_NAME>_s<speaker>_v<phrase variation>_<condition>.wav`

| Component        | Meaning                                                                      |
| ---------------- | ---------------------------------------------------------------------------- |
| Example          | `CALL/CALL_s68_v2_noisy.wav`                                                 |
| `CALL`           | Intent                                                                       |
| `s68`            | Speaker 68, Filipino-English                                                 |
| `v1`, `v2`, `v3` | Phrase variations 1, 2, and 3                                                |
| `clean`, `noisy` | Acoustic condition                                                           |
| Slot value       | Encoded in the containing folder, e.g. `ALARM_6_00AM/ALARM_6_00AM_s68_v1_clean.wav` |
