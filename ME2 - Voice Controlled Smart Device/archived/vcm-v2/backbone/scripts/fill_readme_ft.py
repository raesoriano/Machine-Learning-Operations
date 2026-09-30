#!/usr/bin/env python3
"""Fill the __CMD__/__INT__/__BLANK__/__WER__ placeholders in the READMEs
from the v2 fine-tuned whisper eval reports:
  - backbone/reports/additional_test_whisper_ft_v2.json  (171 held-out clips)
  - backbone/reports/me2_regression_whisper_ft_v2.json   (in-domain guardrail)

NOTE on WER scale: the per-clip `wer` in additional_test_whisper_ft_v2.json
is a FRACTION (0..1+) from wer_breakdown (d[n,m]/n), so mean_wer 10.24
means 1024% WER (the degenerate v1 model). Multiply by 100 for the
percentage shown in the READMEs. The guardrail report's `wer` is also a
fraction.

Usage:
    python backbone/scripts/fill_readme_ft.py
"""
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACK = os.path.dirname(_HERE)
_REPO = os.path.dirname(_BACK)
REPORT = os.path.join(_BACK, "reports", "additional_test_whisper_ft_v2.json")
GUARDRAIL = os.path.join(_BACK, "reports", "me2_regression_whisper_ft_v2.json")
README_TOP = os.path.join(_REPO, "README.md")
README_BACK = os.path.join(_BACK, "README.md")


def main():
    with open(REPORT) as f:
        d = json.load(f)
    with open(GUARDRAIL) as f:
        g = json.load(f)
    repl = {
        "__CMD__": f"{d['command_acc'] * 100:.1f}",
        "__INT__": f"{d['intent_acc'] * 100:.1f}",
        "__BLANK__": f"{d['blank_rate'] * 100:.1f}",
        "__WER__": f"{d['mean_wer'] * 100:.1f}",
        "__GWER__": f"{g['wer'] * 100:.1f}",
        "__GEM__": f"{g['word_exact_match'] * 100:.0f}",
    }
    for path in (README_TOP, README_BACK):
        with open(path) as f:
            txt = f.read()
        for k, v in repl.items():
            txt = txt.replace(k, v)
        with open(path, "w") as f:
            f.write(txt)
    print(json.dumps(repl, indent=1))
    print("READMEs updated")


if __name__ == "__main__":
    main()
