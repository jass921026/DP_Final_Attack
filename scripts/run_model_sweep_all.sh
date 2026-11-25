#!/bin/bash
# 依序對 4 個 victim 架構跑 analysis.p12_model_sweep 的完整 DP sigma 掃描。
# 見 analysis_out/exploration_log_model_sweep.md。
set -e
cd "$(dirname "$0")/.."
COMMON="--n-aux 2000 --n-test 200 --n-bin-aux 1000 --bins 16 --epochs 150 --hidden 1000 1000 --sigmas 0 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2"
for m in itransformer summit mlp gru; do
  echo "=== starting $m at $(date) ===" >> logs/p12_model_sweep_all.log
  .venv/bin/python -m analysis.p12_model_sweep --model "$m" $COMMON --out "p12_model_sweep_$m" >> "logs/p12_model_sweep_$m.log" 2>&1
  echo "=== finished $m at $(date) ===" >> logs/p12_model_sweep_all.log
done
echo "ALL DONE at $(date)" >> logs/p12_model_sweep_all.log
