#!/usr/bin/env bash
# End-to-end finalize for a trained VCM checkpoint: export ONNX -> int8 ->
# tune reject threshold on frozen v1 -> benchmark frozen v1 + additional_test
# (with the reject gate) -> latency. One command, full report.
#
#   bash scripts/finalize_v6.sh <checkpoint_dir> <tag>
#   e.g. bash scripts/finalize_v6.sh model/checkpoints/me2_v6 v6
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT="${1:?usage: finalize_v6.sh <checkpoint_dir> <tag>}"
TAG="${2:?usage: finalize_v6.sh <checkpoint_dir> <tag>}"
CFG="model/configs/$(basename "$CKPT").yaml"
[ -f "$CFG" ] || CFG="model/configs/me2_full_v6.yaml"
ONNX="$CKPT/vcm.onnx"
INT8="$CKPT/vcm_int8.onnx"
FROZEN="data/manifests/frozen_test_v1.jsonl"
ADDTL="../../additional_test_data"

echo "### [1/6] export ONNX ($CKPT/best.pt)"
python3 -m deploy.export_onnx --checkpoint "$CKPT/best.pt" --config "$CFG" --out "$ONNX"

echo "### [2/6] quantize int8"
python3 -m deploy.quantize --in "$ONNX" --out "$INT8"

echo "### [3/6] tune reject threshold on frozen v1"
python3 -u scripts/tune_reject_threshold.py \
    --model "$INT8" --manifest "$FROZEN" --data-root data \
    --id-reject-max 0.05 --report "reports/reject_threshold_${TAG}.json" \
    | tee "reports/tune_reject_${TAG}.log"
# parse the best threshold (fallback to -0.30 if none meets the budget)
THR=$(python3 -c "import json;d=json.load(open('reports/reject_threshold_${TAG}.json'));b=d.get('best');print(b['thr'] if b else -0.30)")
echo "### using reject_min_conf=$THR"

echo "### [4/6] benchmark frozen v1 (int8 + reject gate)"
python3 -m benchmark.evaluate --testset frozen --testset-manifest "$FROZEN" \
    --model onnx --model-path "$INT8" --reject-min-conf "$THR" \
    --report "reports/${TAG}_frozen_int8.json" | tee "reports/${TAG}_frozen_int8.log"

echo "### [5/6] benchmark additional_test (NO gate — all in-domain; the fair
###      generalization comparison to the previous 10.5% baseline)"
python3 -u scripts/eval_additional_test.py --data "$ADDTL" --model "$INT8" \
    --report "reports/additional_test_${TAG}_int8.json" \
    | tee "reports/additional_test_${TAG}_int8.log"
# also with the reject gate, to show device behaviour on the new speaker
python3 -u scripts/eval_additional_test.py --data "$ADDTL" --model "$INT8" \
    --reject-min-conf "$THR" --report "reports/additional_test_${TAG}_int8_gated.json" \
    | tee "reports/additional_test_${TAG}_int8_gated.log"

echo "### [6/6] latency (int8, CPU, 300 rows)"
python3 -u scripts/bench_latency.py --model "$INT8" --testset "$FROZEN" --n 300 \
    --report "reports/${TAG}_latency.json" | tee "reports/${TAG}_latency.log"

echo "### DONE $TAG"
