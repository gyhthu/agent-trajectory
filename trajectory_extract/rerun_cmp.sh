#!/usr/bin/env bash
set +e
cd /home/agent/data_process/trajectory_extract
echo "[START] $(date)"
echo "--- 先等 120s 让当前 v4-pro cooldown 彻底过 ---"
sleep 120
# 补缺：档A chunk1 v4-pro 两轮（拆成 rounds=1 单跑，轮间 sleep 75 防 cooldown）
for r in 1 2; do
  echo "=== 档A chunk1 v4-pro 第 $r 轮 ==="
  DECOMPOSE_ROLE_TAGS=1 python3 gold_decompose_cmp.py --tag A_chunk1_r$r --golds chunk1 --rounds 1
  echo "--- sleep 75 ---"; sleep 75
done
# 补缺：基线 chunk1 第二份对照已有，这里补基线 zym v4-pro（deepseek 重跑无害）
echo "=== 基线 zym v4-pro ==="
DECOMPOSE_ROLE_TAGS=0 python3 gold_decompose_cmp.py --tag base_zym_rerun --golds zym --rounds 1
echo "[ALL DONE] $(date)"
