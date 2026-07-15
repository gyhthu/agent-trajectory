#!/usr/bin/env bash
# 甲·候选召回消融 worker：setsid 后台跑，扛得过对话轮回收；跑完自己 @张耀明 回填结果。
set -uo pipefail
cd "$(dirname "$0")"

CHAT="oc_53b8b620867a189d8dfe502865dfccc5"
YAOMING="ou_adff5621f41381371eec5ca9bb45a9ea"
SEND="/home/agent/lian-server-bot/skills/feishu-plugin/skills/feishu-mention/send.sh"
LOG="/opt/shared/data/task-trajectory/ablation_jia/run_$(date +%m%d_%H%M%S).log"
ROUNDS="${1:-5}"
mkdir -p "$(dirname "$LOG")"

python3 eval_candidate_recall.py --rounds "$ROUNDS" >"$LOG" 2>&1
RC=$?

if [ $RC -ne 0 ]; then
  TAIL=$(tail -25 "$LOG")
  bash "$SEND" -c "$CHAT" -a "$YAOMING" "〔数据处理清洗〕
甲·候选召回消融**跑挂了**（退出码 $RC），没有静默吞。日志尾部：
$TAIL
日志：$LOG"
  exit $RC
fi

# 抽汇总表（从「消融汇总」标题到文末）
SUMMARY=$(awk '/甲·候选召回消融汇总/{f=1} f' "$LOG")
bash "$SEND" -c "$CHAT" -a "$YAOMING" "〔数据处理清洗〕
甲·候选召回消融跑完了（$ROUNDS 轮 ×2条件×2 gold）。判对率对照：

$SUMMARY

完整明细+每轮判定在日志：$LOG
我接着把结论用大白话整理进进展文档。"
