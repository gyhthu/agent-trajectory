#!/usr/bin/env bash
# 档A 前后对比（env 切换，不碰 git）：DECOMPOSE_ROLE_TAGS=0 跑基线、=1 跑档A，同一份工作树。
# chunk1 跑2轮看 v4-pro 稳定性，zym 跑1轮防回归。带429重试由 decompose 内部 retries 兜。
set +e
cd /home/agent/data_process/trajectory_extract || exit 1

echo "=== [1/2] 基线 DECOMPOSE_ROLE_TAGS=0 ==="
DECOMPOSE_ROLE_TAGS=0 python3 gold_decompose_cmp.py --tag base_chunk1 --golds chunk1 --rounds 2
DECOMPOSE_ROLE_TAGS=0 python3 gold_decompose_cmp.py --tag base_zym    --golds zym    --rounds 1

echo "=== [2/2] 档A DECOMPOSE_ROLE_TAGS=1 ==="
DECOMPOSE_ROLE_TAGS=1 python3 gold_decompose_cmp.py --tag A_chunk1 --golds chunk1 --rounds 2
DECOMPOSE_ROLE_TAGS=1 python3 gold_decompose_cmp.py --tag A_zym    --golds zym    --rounds 1

echo "=== 全部完成 ==="
