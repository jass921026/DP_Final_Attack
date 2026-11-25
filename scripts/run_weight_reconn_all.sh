#!/bin/bash
# Balle 原版 weight-ReCoNN（p12_rero_pilot.py 生 shadow/released + p12_reconn.py 訓練 ReCoNN）。
# SUMMIT 已跳過（使用者 2026-11-24 指示）：train_models_batched 的通用 vmap 路徑在 SUMMIT 架構上極度不穩定
# （同一組參數有時 3s/model，有時 40s/model 甚至卡住；多次調大/調小 fixed、chunk、換 mps 都沒有穩定解法，
# 懷疑跟這台機器當下的記憶體/負載波動有關，不是單純調參數能解的問題），只留 imprint closed-form 的 SUMMIT
# 結果（那個已經驗證過沒問題）。iTransformer/MLP/GRU 三個正常跑：
#   1. 非 DP 全尺寸（fixed=500, shadows=1000, targets=200, steps=100；itransformer 重用已有的 cache）
#   2. DP sweep（fixed=300, shadows=400, targets=100；clip=1.0，sigma in {0, 0.1, 0.3, 1.0}）
# 見 analysis_out/exploration_log_model_sweep.md 的後續記錄。
set -e
cd "$(dirname "$0")/.."
DEV="cpu"
LOG=logs/p12_weight_reconn_all.log

echo "=== non-DP full-scale, $(date) ===" >> "$LOG"
for m in mlp gru; do
  tag="${m}_f500_s1000_t200_T100_lr0.03"
  echo "-- pilot $m $(date)" >> "$LOG"
  .venv/bin/python -m analysis.p12_rero_pilot --model "$m" --fixed 500 --shadows 1000 --targets 200 --steps 100 --lr 0.03 --chunk 16 --device $DEV >> "logs/p12_rero_pilot_${m}.log" 2>&1
  echo "-- reconn $m $(date)" >> "$LOG"
  .venv/bin/python -m analysis.p12_reconn --model "$m" --tag "$tag" --fixed 500 --shadows 1000 --targets 200 --hidden 1000 1000 --epochs 200 --device $DEV >> "logs/p12_reconn_${m}.log" 2>&1
done
# itransformer 的非 DP 結果重用已有 cache，補跑一次 reconn 只是為了確保 summary 檔存在、格式一致
echo "-- reconn itransformer (cached shadow/released) $(date)" >> "$LOG"
.venv/bin/python -m analysis.p12_reconn --model itransformer --tag itransformer_f500_s1000_t200_T100_lr0.03 --fixed 500 --shadows 1000 --targets 200 --hidden 1000 1000 --epochs 200 --device $DEV >> logs/p12_reconn_itransformer.log 2>&1

echo "=== DP sweep, $(date) ===" >> "$LOG"
# 2026-11-24 13:30 修正：iTransformer 在 fixed=300/shadows=400/chunk=16 的 DP 路徑（generic vmap，模型維 x
# 樣本維兩層 vmap 疊起來，逐樣本梯度要對 chunk 內每個模型 x 301 筆樣本都個別算一次，100 epoch 下來計算量
# 爆炸）跑了 2 小時連第一個 100-model checkpoint 都沒印出來，跟系統負載無關，是這條路徑本身對這個規模不現實。
# MLP 有專屬快速 DP 路徑（_mlp_dp_grads，不受影響，照全尺寸跑）；iTransformer/GRU 砍到 shadows=15/targets=5
# （20 個模型/點 x 4 點），換一個能跑完、雖然樣本少但至少是真實 DP 數字的結果，而不是完全沒有。
for m in itransformer mlp gru; do
  if [ "$m" = "mlp" ]; then FIXED=300; SHAD=400; TARG=100; CHUNK=16; else FIXED=100; SHAD=15; TARG=5; CHUNK=5; fi
  for sig in 0 0.1 0.3 1.0; do
    tag="${m}_f${FIXED}_s${SHAD}_t${TARG}_T100_lr0.03_dpC1.0_sig${sig}"
    echo "-- pilot $m sigma=$sig $(date)" >> "$LOG"
    .venv/bin/python -m analysis.p12_rero_pilot --model "$m" --fixed $FIXED --shadows $SHAD --targets $TARG --steps 100 --lr 0.03 --clip 1.0 --sigma "$sig" --chunk $CHUNK --device $DEV >> "logs/p12_rero_pilot_${m}_dp.log" 2>&1
    echo "-- reconn $m sigma=$sig $(date)" >> "$LOG"
    .venv/bin/python -m analysis.p12_reconn --model "$m" --tag "$tag" --fixed $FIXED --shadows $SHAD --targets $TARG --hidden 512 512 --epochs 150 --device $DEV >> "logs/p12_reconn_${m}_dp.log" 2>&1
  done
done
echo "ALL DONE $(date)" >> "$LOG"
