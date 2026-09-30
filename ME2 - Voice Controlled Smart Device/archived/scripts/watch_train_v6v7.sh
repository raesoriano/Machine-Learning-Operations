#!/bin/bash
# Watch v6+v7 ASR training to completion, then finalize both checkpoints
# (export ONNX -> int8 -> reject-threshold tune -> frozen bench ->
#  additional_test no-gate + gated -> latency).
#
#   bash scripts/watch_train_v6v7.sh <v6_pid> <v7_pid>
cd "/home/ron.andrei.soriano/sandbox/Machine-Learning-Operations/ME2 - Voice Controlled Smart Device" || exit 1
V6="${1:?v6 pid}"
V7="${2:?v7 pid}"
echo "WATCH_START $(date)"
for i in $(seq 1 180); do
  if ! kill -0 "$V6" 2>/dev/null && ! kill -0 "$V7" 2>/dev/null; then
    echo "BOTH_EXITED after ~$((i*2)) min checks $(date)"
    break
  fi
  sleep 120
done
if kill -0 "$V6" 2>/dev/null || kill -0 "$V7" 2>/dev/null; then
  echo "TIMEOUT: training still running after 6h"
  exit 1
fi
echo "=== train tails ==="
tail -5 reports/train_v6.log
tail -5 reports/train_v7.log
# preserve the partial-run (ep100 backup) v6 reports before finalize overwrites
for f in additional_test_v6_int8 additional_test_v6_int8_gated v6_frozen_int8 v6_latency reject_threshold_v6; do
  [ -f "reports/${f}.json" ] && cp "reports/${f}.json" "reports/${f}_partial.json"
done
echo "=== FINALIZE v6 ==="
bash scripts/finalize_v6.sh model/checkpoints/me2_v6 v6 || echo "FINALIZE_V6_FAILED"
echo "=== FINALIZE v7 ==="
bash scripts/finalize_v6.sh model/checkpoints/me2_v7 v7 || echo "FINALIZE_V7_FAILED"
echo "WATCH_DONE $(date)"
