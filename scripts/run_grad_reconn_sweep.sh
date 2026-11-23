#!/bin/bash
# 梯度輸入 ReCoNN 的全掃描（8 小時探索預算）。Tier A/B/C 依序跑完，每個 tier 結束後印出一個分隔線方便在 log 裡找進度。
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
OUT=p12_grad_reconn

echo "=== TIER A: sigma sweep, bins=16, n_aux=2000, epochs=150, seeds 0,1,2 ==="
for seed in 0 1 2; do
  for sigma in 0 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2; do
    echo "--- TIER A seed=$seed sigma=$sigma ---"
    $PY -m analysis.p12_grad_reconn --n-aux 2000 --n-test 200 --n-bin-aux 1000 --bins 16 --epochs 150 \
      --sigma "$sigma" --seed "$seed" --out "$OUT"
  done
done

echo "=== TIER B: bins ablation, seed=0, n_aux=2000, epochs=150 ==="
for bins in 8 32; do
  for sigma in 0 1e-3 3e-3 1e-2; do
    echo "--- TIER B bins=$bins sigma=$sigma ---"
    $PY -m analysis.p12_grad_reconn --n-aux 2000 --n-test 200 --n-bin-aux 1000 --bins "$bins" --epochs 150 \
      --sigma "$sigma" --seed 0 --out "$OUT"
  done
done

echo "=== TIER C: bigger aux pool, seed=0, bins=16, n_aux=4000, epochs=250 ==="
for sigma in 0 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2; do
  echo "--- TIER C sigma=$sigma ---"
  $PY -m analysis.p12_grad_reconn --n-aux 4000 --n-test 200 --n-bin-aux 1000 --bins 16 --epochs 250 \
    --sigma "$sigma" --seed 0 --out "$OUT"
done

echo "=== SWEEP DONE ==="
