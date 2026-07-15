#!/bin/bash
# 增量轨迹切分·定时驱动（供 crontab 调用，也可手动跑）
#
# 做什么：读 trajectory_groups.txt 的群清单，逐群调 incremental_segment.py——
#   已结束的老任务封存复用、只重切活动尾巴和被回复唤醒的老任务（省钱、不抖）。
# fail-loud：单个群出错只记错并继续下一个群，全部跑完若有任何群失败则整体退非零，
#   让 cron 日志能一眼看出问题（不静默吞）。
#
# 可调 env（都有默认值）：
#   TRAJ_ENV_FILE     凭证 .env（FEISHU_APP_ID/SECRET 等），默认 team-agent-server/.env
#   TRAJ_GROUPS_FILE  群清单文件，默认与本脚本同目录的 trajectory_groups.txt
#   LLM_SEGMENT_MODEL / LLM_DECOMPOSE_MODEL 等沿用代码默认（v4-pro / deepseek）
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${TRAJ_ENV_FILE:-/home/agent/team-agent-server/.env}"
GROUPS_FILE="${TRAJ_GROUPS_FILE:-$HERE/trajectory_groups.txt}"

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

if [ -f "$ENV_FILE" ]; then
  set -a; . "$ENV_FILE"; set +a
else
  echo "[$(ts)] ⚠️ 凭证文件不存在：$ENV_FILE（飞书历史拉取可能失败）" >&2
fi

if [ ! -f "$GROUPS_FILE" ]; then
  echo "[$(ts)] ❌ 群清单不存在：$GROUPS_FILE" >&2
  exit 2
fi

# 读清单：去行内 # 注释、去首尾空白、跳空行。
# ⚠️ 变量名别用 GROUPS——它是 bash 保留特殊变量（存当前用户的组 ID = `id -G`），
#    对它赋值会被忽略，曾导致清单被 `id -G`（1001 988 1010）顶替。故用 GROUP_LIST。
GROUP_LIST=()
while IFS= read -r _line || [ -n "$_line" ]; do
  _line="${_line%%#*}"                              # 去行内注释
  _line="${_line#"${_line%%[![:space:]]*}"}"        # 去左空白
  _line="${_line%"${_line##*[![:space:]]}"}"        # 去右空白
  [ -n "$_line" ] && GROUP_LIST+=("$_line")
done < "$GROUPS_FILE"

if [ "${#GROUP_LIST[@]}" -eq 0 ]; then
  echo "[$(ts)] ❌ 群清单为空，无群可处理：$GROUPS_FILE" >&2
  exit 2
fi

echo "[$(ts)] 开始增量切分，共 ${#GROUP_LIST[@]} 个群"
fail=0
for g in "${GROUP_LIST[@]}"; do
  echo "[$(ts)] === 群 $g ==="
  if python3 "$HERE/incremental_segment.py" --group "$g"; then
    echo "[$(ts)] ✅ $g 完成"
  else
    rc=$?
    echo "[$(ts)] ❌ $g 失败（exit=$rc），继续下一个群" >&2
    fail=1
  fi
done

echo "[$(ts)] 全部跑完$([ "$fail" -eq 0 ] && echo "，均成功" || echo "，有群失败↑")"

# 切完刷新任务状态面板（只读视图，扫所有群 state 汇总；失败不影响切分主流程的退出码）
if python3 "$HERE/task_panel.py" >/dev/null 2>&1; then
  echo "[$(ts)] 📋 任务面板已刷新：/opt/shared/data/task-trajectory/task_panel.md"
else
  echo "[$(ts)] ⚠️ 任务面板刷新失败（不影响切分结果）" >&2
fi

exit "$fail"
